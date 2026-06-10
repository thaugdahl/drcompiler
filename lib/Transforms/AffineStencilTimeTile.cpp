//===- AffineStencilTimeTile.cpp - skewed time-tiling for stencils --------===//
//
// Time-tiles ping-pong stencils (PolyBench jacobi-1d/2d, heat-3d at -O0):
//
//     for t { nest1: B <- f(A);  nest2: A <- g(B) }
//
// where both nests are perfect constant-bound bands over the same space and
// every load offset is within +/-1 of the store point (halo 1).  One time
// step streams both whole arrays; once they exceed the cache, every step
// re-misses everything.  Tiling t alone is illegal (the halo couples
// neighbouring points across steps) -- the classical fix is skewing.
//
// Schedule derivation (the legality argument, kept here on purpose):
// give each half-step a virtual time tau = 2t + phase (phase 0 = nest1,
// 1 = nest2) and skew every space dim by tau: i'_k = i_k + tau.  Every
// dependence (an update at tau reads points at tau-1 within halo 1) then
// has distance vector (Delta tau, Delta i'_k) = (1, 1 + delta_k) with
// delta_k in {-1,0,1}, i.e. ALL components >= 0: the (tau, i'_1..i'_d)
// band is fully permutable, so rectangular tiling of the skewed space with
// lexicographic tile order is legal (Irigoin/Triolet, Wolf/Lam).
//
// The tile body is emitted in ORIGINAL coordinates -- the skew only shapes
// the per-(tile, t) iteration windows, which become affine max/min bounds:
//
//   for tt step Tt:                       // time tile
//     for ii_k step Ts (skewed range):    // space tiles, per dim
//       for t = tt .. min(tt+Tt, T):
//         nest1, dim k in [max(lo_k, ii_k - 2t),   min(hi_k, ii_k+Ts - 2t))
//         nest2, dim k in [max(lo_k, ii_k - 2t-1), min(hi_k, ii_k+Ts - 2t-1))
//
// (windows = the tile's slice of skewed space translated back by tau, so
// for fixed tau the ii_k windows partition the space exactly: every (t, i)
// body instance runs exactly once, whole -- no FP op is reordered within a
// point update, and per-point update order is preserved, so the result is
// BIT-IDENTICAL.)  Bounds hang off real IVs via maps (no affine.apply
// between loops): the shape is unroll-jam-proof by construction.
//
//===----------------------------------------------------------------------===//

#include "drcompiler/Transforms/AffineStencilTimeTile.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
#define GEN_PASS_DEF_DRAFFINESTENCILTIMETILEPASS
#include "drcompiler/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "dr-affine-stencil-time-tile"

using namespace mlir;
using affine::AffineForOp;
using affine::AffineLoadOp;
using affine::AffineStoreOp;

namespace {

/// A matched ping-pong stencil step: one perfect constant-bound space nest.
struct SpaceNest {
  AffineForOp root;
  SmallVector<AffineForOp, 3> loops; // outer..inner
  Value readArr, writeArr;
};

/// If `forOp` roots a perfect nest of constant-bound unit-step loops whose
/// innermost body is straight-line loads/stores, fill `nest` and return true.
static bool matchSpaceNest(AffineForOp forOp, SpaceNest &nest) {
  nest.root = forOp;
  nest.loops.clear();
  AffineForOp cur = forOp;
  while (true) {
    if (!cur.hasConstantLowerBound() || !cur.hasConstantUpperBound() ||
        cur.getStepAsInt() != 1)
      return false;
    nest.loops.push_back(cur);
    if (nest.loops.size() > 3)
      return false;
    AffineForOp child;
    bool sawNonLoop = false;
    for (Operation &op : cur.getBody()->without_terminator()) {
      if (auto f = dyn_cast<AffineForOp>(&op)) {
        if (child)
          return false; // two sibling loops
        child = f;
      } else {
        sawNonLoop = true;
      }
    }
    if (!child) {
      // Innermost: the non-loop ops are the stencil body, validated by
      // classifyBody.
      return true;
    }
    if (sawNonLoop)
      return false; // imperfect level
    cur = child;
  }
}

/// Check the innermost body of `nest`: every subscript of every access must
/// be `IV_k + c` with |c| <= 1 (c == 0 for stores), loads all from one
/// memref, stores all to one DIFFERENT memref, nothing else effectful.
/// Records the two arrays.
static bool classifyBody(SpaceNest &nest) {
  Block *body = nest.loops.back().getBody();
  Value readArr, writeArr;
  auto subscriptsOk = [&](AffineMap map, ValueRange ops, bool isStore) {
    if (map.getNumResults() != nest.loops.size() ||
        map.getNumSymbols() != 0 || ops.size() != map.getNumDims())
      return false;
    for (unsigned d = 0, e = map.getNumResults(); d < e; ++d) {
      AffineExpr expr = map.getResult(d);
      int64_t c = 0;
      AffineExpr dimExpr = expr;
      if (auto bin = dyn_cast<AffineBinaryOpExpr>(expr)) {
        if (bin.getKind() != AffineExprKind::Add)
          return false;
        auto cst = dyn_cast<AffineConstantExpr>(bin.getRHS());
        if (!cst)
          return false;
        c = cst.getValue();
        dimExpr = bin.getLHS();
      }
      auto dim = dyn_cast<AffineDimExpr>(dimExpr);
      if (!dim)
        return false;
      // Subscript d must use space IV d (same order as the nest).
      if (ops[dim.getPosition()] != nest.loops[d].getInductionVar())
        return false;
      if (isStore ? c != 0 : (c < -1 || c > 1))
        return false;
    }
    return true;
  };
  for (Operation &op : body->without_terminator()) {
    if (auto ld = dyn_cast<AffineLoadOp>(&op)) {
      if (!subscriptsOk(ld.getAffineMap(), ld.getMapOperands(), false))
        return false;
      if (readArr && ld.getMemRef() != readArr)
        return false;
      readArr = ld.getMemRef();
    } else if (auto st = dyn_cast<AffineStoreOp>(&op)) {
      if (!subscriptsOk(st.getAffineMap(), st.getMapOperands(), true))
        return false;
      if (writeArr && st.getMemRef() != writeArr)
        return false;
      writeArr = st.getMemRef();
    } else if (!isMemoryEffectFree(&op)) {
      return false;
    }
  }
  if (!readArr || !writeArr || readArr == writeArr)
    return false;
  nest.readArr = readArr;
  nest.writeArr = writeArr;
  return true;
}

class DrAffineStencilTimeTilePass final
    : public impl::DrAffineStencilTimeTilePassBase<
          DrAffineStencilTimeTilePass> {
public:
  using DrAffineStencilTimeTilePassBase<
      DrAffineStencilTimeTilePass>::DrAffineStencilTimeTilePassBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    SmallVector<AffineForOp> tLoops;
    func.walk([&](AffineForOp t) {
      if (t->getParentOfType<AffineForOp>())
        return; // only top-level time loops
      SmallVector<AffineForOp, 2> nests;
      for (Operation &op : t.getBody()->without_terminator()) {
        if (auto f = dyn_cast<AffineForOp>(&op))
          nests.push_back(f);
        else if (!isMemoryEffectFree(&op))
          return;
      }
      if (nests.size() == 2)
        tLoops.push_back(t);
    });

    for (AffineForOp t : tLoops)
      (void)timeTile(t);
  }

private:
  LogicalResult timeTile(AffineForOp t) {
    if (!t.hasConstantLowerBound() || !t.hasConstantUpperBound() ||
        t.getStepAsInt() != 1)
      return failure();
    SmallVector<AffineForOp, 2> nestRoots;
    for (Operation &op : t.getBody()->without_terminator())
      nestRoots.push_back(cast<AffineForOp>(&op));

    SpaceNest n1, n2;
    if (!matchSpaceNest(nestRoots[0], n1) || !classifyBody(n1) ||
        !matchSpaceNest(nestRoots[1], n2) || !classifyBody(n2))
      return failure();
    // Ping-pong: nest2 consumes what nest1 produced and refreshes nest1's
    // input.  Identical space bands (same depth + bounds).
    if (n1.writeArr != n2.readArr || n1.readArr != n2.writeArr)
      return failure();
    unsigned d = n1.loops.size();
    if (n2.loops.size() != d)
      return failure();
    SmallVector<int64_t, 3> slo(d), shi(d);
    for (unsigned k = 0; k < d; ++k) {
      slo[k] = n1.loops[k].getConstantLowerBound();
      shi[k] = n1.loops[k].getConstantUpperBound();
      if (n2.loops[k].getConstantLowerBound() != slo[k] ||
          n2.loops[k].getConstantUpperBound() != shi[k])
        return failure();
      if (shi[k] <= slo[k])
        return failure();
    }

    // Profitability: time-tiling pays only when one step's working set
    // (both arrays) overflows the cache share we can count on; below that
    // the t loop already runs cache-resident and the skewed windows only
    // add min/max overhead (Polly's jacobi-1d 0.34x is this mistake).
    if (!forceTile) {
      int64_t elemBytes = 8;
      if (auto mt = dyn_cast<MemRefType>(n1.readArr.getType()))
        if (mt.getElementType().isIntOrFloat())
          elemBytes = std::max<int64_t>(
              1, (int64_t)mt.getElementType().getIntOrFloatBitWidth() / 8);
      int64_t pts = 1;
      for (unsigned k = 0; k < d; ++k)
        pts *= shi[k] - slo[k];
      int64_t stepBytes = 2 * pts * elemBytes;
      int64_t effLLC = (int64_t)l3Size / (int64_t)(llcSharers ? llcSharers : 1u);
      if (stepBytes <= effLLC)
        return failure();
    }

    int64_t tlo = t.getConstantLowerBound(), thi = t.getConstantUpperBound();
    // tile-t = 0: measured XL optima (plateaus): d=2 -> 128 (jacobi-2d
    // 2.9x, flat 64..128), d=3 -> 24 (heat-3d 2.08x; 32 already degrades
    // because the skew slack 2*Tt eats the whole cache-derived strip).
    int64_t Tt = tileT ? std::max<int64_t>(1, tileT)
                       : (d == 3 ? 24 : (d == 2 ? 128 : 32));
    // tile-s = 0: derive the strip width from the cache so one tile's
    // working set (both arrays over a (Ts + 2*Tt)^d skew-extended box)
    // stays within half the effective LLC.
    int64_t Ts = tileS;
    if (Ts == 0) {
      int64_t effLLC =
          (int64_t)l3Size / (int64_t)(llcSharers ? llcSharers : 1u);
      double box = (double)effLLC / 2.0 / 16.0; // 2 arrays x 8 B
      double side = std::pow(box, 1.0 / (double)d);
      Ts = (int64_t)side - 2 * Tt;
    }
    Ts = std::max<int64_t>(8, Ts);

    MLIRContext *ctx = t.getContext();
    Location loc = t.getLoc();
    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(t);

    AffineExpr d0 = getAffineDimExpr(0, ctx), d1 = getAffineDimExpr(1, ctx);

    // for tt = tlo .. thi step Tt
    auto ttL = rewriter.create<AffineForOp>(loc, tlo, thi, Tt);
    rewriter.setInsertionPointToStart(ttL.getBody());
    // Space tile loops over the BAND-LOCAL skewed ranges
    // [slo_k + 2*tt, shi_k + 2*(tt + Tt)): within one tt band the skew
    // offset only spans 2*Tt, so this covers every window the band can
    // reach while avoiding the (whole-2T-extension)^d grid of empty tiles
    // that otherwise dominates at d >= 3.
    SmallVector<AffineForOp, 3> iiLs;
    for (unsigned k = 0; k < d; ++k) {
      AffineMap iiLb =
          AffineMap::get(1, 0, getAffineConstantExpr(slo[k], ctx) + 2 * d0);
      AffineMap iiUb = AffineMap::get(
          1, 0, getAffineConstantExpr(shi[k], ctx) + 2 * d0 + 2 * Tt);
      auto iiL = rewriter.create<AffineForOp>(
          loc, ValueRange{ttL.getInductionVar()}, iiLb,
          ValueRange{ttL.getInductionVar()}, iiUb, Ts);
      rewriter.setInsertionPointToStart(iiL.getBody());
      iiLs.push_back(iiL);
    }
    // for t' = tt .. min(tt+Tt, thi)
    AffineMap tLb = AffineMap::get(1, 0, d0);
    AffineMap tUb = AffineMap::get(1, 0, {d0 + Tt, getAffineConstantExpr(thi, ctx)}, ctx);
    auto tL = rewriter.create<AffineForOp>(
        loc, ValueRange{ttL.getInductionVar()}, tLb,
        ValueRange{ttL.getInductionVar()}, tUb, 1);
    rewriter.setInsertionPointToStart(tL.getBody());

    // Clone one phase nest with windowed bounds; phase 0 = nest1 (tau = 2t),
    // phase 1 = nest2 (tau = 2t + 1).
    auto emitPhase = [&](SpaceNest &n, int64_t phase) {
      auto cloned = cast<AffineForOp>(rewriter.clone(*n.root.getOperation()));
      // Rebind each level's bounds:
      //   lb = max(slo_k, ii_k - 2t - phase)
      //   ub = min(shi_k, ii_k + Ts - 2t - phase)
      AffineForOp cur = cloned;
      for (unsigned k = 0; k < d; ++k) {
        AffineMap lb = AffineMap::get(
            2, 0, {getAffineConstantExpr(slo[k], ctx), d0 - 2 * d1 - phase},
            ctx);
        AffineMap ub = AffineMap::get(
            2, 0,
            {getAffineConstantExpr(shi[k], ctx), d0 + Ts - 2 * d1 - phase},
            ctx);
        SmallVector<Value, 2> ops{iiLs[k].getInductionVar(),
                                  tL.getInductionVar()};
        cur.setLowerBound(ops, lb);
        cur.setUpperBound(ops, ub);
        if (k + 1 < d) {
          for (Operation &op : cur.getBody()->without_terminator())
            if (auto f = dyn_cast<AffineForOp>(&op)) {
              cur = f;
              break;
            }
        }
      }
      // Re-map any use of the original t IV inside the clone.
      cloned->walk([&](Operation *op) {
        for (OpOperand &o : op->getOpOperands())
          if (o.get() == t.getInductionVar())
            o.set(tL.getInductionVar());
      });
      rewriter.setInsertionPointAfter(cloned);
    };
    emitPhase(n1, 0);
    emitPhase(n2, 1);

    rewriter.eraseOp(t);
    return success();
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createDrAffineStencilTimeTilePass() {
  return std::make_unique<DrAffineStencilTimeTilePass>();
}
} // namespace mlir
