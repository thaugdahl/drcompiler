//===- LowerKrnlGlobal.cpp - krnl.global -> memref.global (krnl-free) ---===//
//
// onnx-mlir's monolithic `convert-krnl-to-llvm` rejects external parallel
// constructs (omp / scf.parallel), so the whole-kernel SPMD path (par->omp)
// cannot lower through it.  This pass strips the only krnl ops that survive
// into the affine stage so the rest of the kernel can be lowered to LLVM by
// HOST mlir-opt (keeping omp intact, then --convert-openmp-to-llvm):
//
//   "krnl.global"() {name, shape, value} : () -> memref<...>
//       =>  memref.global "private" constant @<name> : memref<...> = <value>
//           %g = memref.get_global @<name> : memref<...>
//   "krnl.entry_point"() ...   =>  erased (the run_main_graph wrapper is
//       replaced by a hand-written C harness calling _mlir_ciface_main_graph).
//
// krnl ops are unregistered here (dr-opt runs with -allow-unregistered-dialect),
// so they are matched by op name.  Diagnostic / opt-in; no effect on modules
// without krnl ops.
//
//===----------------------------------------------------------------------===//

#include "drcompiler/Transforms/LowerKrnlGlobal.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
#define GEN_PASS_DEF_LOWERKRNLGLOBALPASS
#include "drcompiler/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

/// A contiguous memref viewed as 1-D (offset 0, unit stride) for a flat copy.
/// Requires a static, contiguous source (true for the alloc'd activation
/// buffers krnl.memcpy operates on).
static Value flatten1D(OpBuilder &b, Location loc, Value memref) {
  auto ty = cast<MemRefType>(memref.getType());
  int64_t n = 1;
  for (int64_t d : ty.getShape())
    n *= d;
  auto flatTy = MemRefType::get({n}, ty.getElementType());
  return b.create<memref::ReinterpretCastOp>(
      loc, flatTy, memref, /*offset=*/b.getIndexAttr(0),
      /*sizes=*/ArrayRef<OpFoldResult>{b.getIndexAttr(n)},
      /*strides=*/ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
}

/// Row-major contiguous strides of a static memref, or empty if it isn't a
/// static identity-layout memref.
static SmallVector<int64_t> contiguousStrides(MemRefType ty) {
  if (!ty.hasStaticShape())
    return {};
  int64_t off;
  SmallVector<int64_t> strides;
  if (failed(ty.getStridesAndOffset(strides, off)) || off != 0)
    return {};
  // Require the canonical row-major contiguous layout (the alloc'd activation
  // buffers krnl.memcpy operates on always have it).
  SmallVector<int64_t> expect(ty.getRank());
  int64_t acc = 1;
  for (int d = ty.getRank() - 1; d >= 0; --d) {
    expect[d] = acc;
    acc *= ty.getShape()[d];
  }
  return strides == expect ? expect : SmallVector<int64_t>{};
}

/// Decompose a single-result affine map into per-input-dim integer coefficients
/// (`coeffs[k]` = the coefficient of input dim k) and a constant term.  Returns
/// false on anything that isn't a pure linear combination of dims (symbols,
/// mod/floordiv, etc.) -- the caller then falls back to the flat copy.
static bool linearCoeffs(AffineExpr e, unsigned numDims,
                         SmallVectorImpl<int64_t> &coeffs, int64_t &constTerm) {
  if (auto c = dyn_cast<AffineConstantExpr>(e)) {
    constTerm += c.getValue();
    return true;
  }
  if (auto d = dyn_cast<AffineDimExpr>(e)) {
    coeffs[d.getPosition()] += 1;
    return true;
  }
  auto bin = dyn_cast<AffineBinaryOpExpr>(e);
  if (!bin)
    return false;
  if (bin.getKind() == AffineExprKind::Add)
    return linearCoeffs(bin.getLHS(), numDims, coeffs, constTerm) &&
           linearCoeffs(bin.getRHS(), numDims, coeffs, constTerm);
  if (bin.getKind() == AffineExprKind::Mul) {
    auto dim = dyn_cast<AffineDimExpr>(bin.getLHS());
    auto cst = dyn_cast<AffineConstantExpr>(bin.getRHS());
    if (dim && cst) {
      coeffs[dim.getPosition()] += cst.getValue();
      return true;
    }
  }
  return false; // not a linear dim term
}

/// Build a MULTI-DIMENSIONAL affine access `mem[c_0, ..., c_{n-2}, iv]` that
/// addresses the same element as flat `mem[offset + iv]`, by delinearizing the
/// affine-apply `offset` against `mem`'s contiguous strides: each offset term
/// `operand_k * stride_d` places `operand_k` at memref dimension d; the copy IV
/// fills the contiguous innermost dim.  Per-dimension subscripts let the affine
/// dependence test prove per-shard owner-disjointness (a transpose is disjoint
/// on every axis), so the band shards into a par.forall instead of falling to
/// par.critical.  Returns false (caller falls back to the flat copy, which
/// stays critical -- sound) unless the pattern matches exactly: static
/// row-major memref, `size` == innermost extent, offset a pure linear
/// affine.apply whose coefficients are distinct non-innermost strides.
static bool buildDelinAccess(Value mem, Value offset, int64_t size, Value iv,
                             AffineMap &accMap, SmallVectorImpl<Value> &operands) {
  auto ty = dyn_cast<MemRefType>(mem.getType());
  if (!ty)
    return false;
  SmallVector<int64_t> strides = contiguousStrides(ty);
  unsigned n = ty.getRank();
  if (strides.empty() || n < 2)
    return false;
  if (size != ty.getShape()[n - 1]) // the copy must fill the innermost dim
    return false;
  auto apply = offset.getDefiningOp<affine::AffineApplyOp>();
  if (!apply || apply.getAffineMap().getNumResults() != 1 ||
      apply.getAffineMap().getNumSymbols() != 0)
    return false;
  unsigned m = apply.getAffineMap().getNumDims();
  SmallVector<int64_t> coeffs(m, 0);
  int64_t constTerm = 0;
  if (!linearCoeffs(apply.getAffineMap().getResult(0), m, coeffs, constTerm))
    return false;
  if (constTerm != 0) // a non-coordinate-aligned start: don't handle
    return false;
  // For each leading memref dim, find the unique offset operand whose
  // coefficient equals that dim's stride.
  SmallVector<AffineExpr> results(n);
  MLIRContext *ctx = mem.getContext();
  for (unsigned d = 0; d + 1 < n; ++d) {
    int found = -1;
    for (unsigned k = 0; k < m; ++k)
      if (coeffs[k] == strides[d]) {
        if (found != -1)
          return false; // ambiguous (two operands share a stride)
        found = (int)k;
      }
    results[d] = found >= 0 ? getAffineDimExpr(found, ctx)
                            : getAffineConstantExpr(0, ctx);
  }
  // Every offset operand must map to some leading dim (else the offset reaches
  // into the innermost/contiguous range and a flat fill is wrong).
  for (unsigned k = 0; k < m; ++k) {
    if (coeffs[k] == 0)
      continue;
    bool used = false;
    for (unsigned d = 0; d + 1 < n; ++d)
      if (coeffs[k] == strides[d])
        used = true;
    if (!used)
      return false;
  }
  results[n - 1] = getAffineDimExpr(m, ctx); // copy IV fills the innermost dim
  accMap = AffineMap::get(/*dimCount=*/m + 1, /*symbolCount=*/0, results, ctx);
  operands.assign(apply.getMapOperands().begin(), apply.getMapOperands().end());
  operands.push_back(iv);
  return true;
}

/// "krnl.memcpy"(dst, src, size, [destOff, srcOff]) -> a flat AFFINE copy:
///   affine.for %i = 0 to size:
///     dstFlat[destOff + i] = srcFlat[srcOff + i]
///
/// Emitting an affine.for (not scf.for) with affine.load/store keeps the band
/// FULLY AFFINE so par-spmd-perband's oracle can analyze it: the dominant
/// openai-gpt attention head reshape/transpose bands (1 gather + 36 reshapes,
/// PARALLEL_SPMD_SPEC.md §11.20) lower through krnl.memcpy and were the SPMD
/// serial floor (37 par.critical bands).  With an affine body, Tier-1
/// checkMemrefAccessDependence PROVES per-shard owner-disjointness (distinct
/// src/dst allocation roots; injective reshape index) and the band shards into
/// a par.forall.  An overlapping / in-place reshape that is NOT provably
/// disjoint returns Carried -> the band correctly STAYS par.critical (sound:
/// the rewrite changes only the copy's loop form, never its memory semantics).
///
/// The index map is `(d0, d1) -> (d0 + d1)` over the IV and the offset, both as
/// affine DIMS: the offsets are affine.apply results of the enclosing band IVs
/// (valid affine dims), so the composed access is a valid affine expression the
/// dependence machinery composes through.
static void lowerMemcpy(Operation *op) {
  OpBuilder b(op);
  Location loc = op->getLoc();
  Value dst = op->getOperand(0), src = op->getOperand(1);
  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value destOff = op->getNumOperands() > 3 ? op->getOperand(3) : c0;
  Value srcOff = op->getNumOperands() > 4 ? op->getOperand(4) : c0;

  // Constant copy length (the c64 head-reshape copies); -1 if dynamic.
  int64_t sizeC = -1;
  if (auto c = op->getOperand(2).getDefiningOp<arith::ConstantOp>())
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
      sizeC = ia.getInt();

  // Preferred: a MULTI-DIMENSIONAL affine copy on the original memrefs, so the
  // dependence test can prove per-shard disjointness and the band shards into a
  // par.forall (PARALLEL_SPMD_SPEC.md §11.20).  Needs a constant size matching
  // both innermost extents and offsets that delinearize cleanly.
  if (sizeC > 0) {
    auto srcTy = dyn_cast<MemRefType>(src.getType());
    auto dstTy = dyn_cast<MemRefType>(dst.getType());
    if (srcTy && dstTy) {
      auto loop = b.create<affine::AffineForOp>(loc, 0, sizeC, 1);
      OpBuilder lb(loop.getBody(), loop.getBody()->begin());
      Value i = loop.getInductionVar();
      AffineMap sMap, dMap;
      SmallVector<Value> sOps, dOps;
      if (buildDelinAccess(src, srcOff, sizeC, i, sMap, sOps) &&
          buildDelinAccess(dst, destOff, sizeC, i, dMap, dOps)) {
        Value v = lb.create<affine::AffineLoadOp>(loc, src, sMap, sOps);
        lb.create<affine::AffineStoreOp>(loc, v, dst, dMap, dOps);
        op->erase();
        return;
      }
      loop.erase(); // delinearization didn't match: fall through to flat copy
    }
  }

  // Fallback: the flat 1-D copy (the known-correct form for any pattern the
  // multi-D path declines -- it stays par.critical, which is sound).
  Value dstFlat = flatten1D(b, loc, dst), srcFlat = flatten1D(b, loc, src);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value size = op->getOperand(2);
  if (!size.getType().isIndex())
    size = b.create<arith::IndexCastOp>(loc, b.getIndexType(), size);
  auto floop = b.create<scf::ForOp>(loc, c0, size, c1);
  OpBuilder lb(floop.getBody(), floop.getBody()->begin());
  Value i = floop.getInductionVar();
  Value si = lb.create<arith::AddIOp>(loc, srcOff, i);
  Value di = lb.create<arith::AddIOp>(loc, destOff, i);
  Value v = lb.create<memref::LoadOp>(loc, srcFlat, si);
  lb.create<memref::StoreOp>(loc, v, dstFlat, di);
  op->erase();
}

struct LowerKrnlGlobalPass
    : public impl::LowerKrnlGlobalPassBase<LowerKrnlGlobalPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder modBuilder(module.getBody(), module.getBody()->begin());
    unsigned counter = 0;

    SmallVector<Operation *> globals, entryPoints, memcpys;
    module.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "krnl.global")
        globals.push_back(op);
      else if (name == "krnl.entry_point")
        entryPoints.push_back(op);
      else if (name == "krnl.memcpy")
        memcpys.push_back(op);
    });
    for (Operation *op : memcpys)
      lowerMemcpy(op);

    for (Operation *op : globals) {
      auto memrefTy = dyn_cast<MemRefType>(op->getResult(0).getType());
      Attribute value = op->getAttr("value");
      if (!memrefTy || !value) // not the shape we expect: leave it (will error
        continue;              // downstream, surfacing the unhandled case)

      // A unique, valid symbol name (prefer the krnl 'name' attr).
      std::string sym = "krnlg_" + std::to_string(counter++);
      if (auto nameAttr = op->getAttrOfType<StringAttr>("name"))
        sym = "krnlg_" + nameAttr.getValue().str();

      modBuilder.setInsertionPointToStart(module.getBody());
      modBuilder.create<memref::GlobalOp>(
          op->getLoc(), modBuilder.getStringAttr(sym),
          /*sym_visibility=*/modBuilder.getStringAttr("private"),
          /*type=*/TypeAttr::get(memrefTy),
          /*initial_value=*/value,
          /*constant=*/modBuilder.getUnitAttr(),
          /*alignment=*/IntegerAttr());

      OpBuilder b(op);
      auto get = b.create<memref::GetGlobalOp>(op->getLoc(), memrefTy, sym);
      op->getResult(0).replaceAllUsesWith(get.getResult());
      op->erase();
    }
    for (Operation *op : entryPoints)
      op->erase();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createLowerKrnlGlobalPass() {
  return std::make_unique<LowerKrnlGlobalPass>();
}
