//===- RematKernel.cpp - Rematerialization analysis & cloning kernel --===//
//
//===----------------------------------------------------------------------===//

#include "drcompiler/Transforms/DataRecomputation/RematKernel.h"

#include "drcompiler/Transforms/Utils/MemrefBaseAnalysis.h"
#include "drcompiler/Transforms/Utils/OpDispatchUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <algorithm>
#include <functional>

using drcompiler::collectBaseMemrefs;
using drcompiler::getStoredValue;

namespace dr {

// Forward declarations.
static std::optional<int64_t> tryFoldAffineValueLocal(mlir::Value v);

bool matchEnclosingForNests(
    mlir::Operation *producerOp, mlir::Operation *consumerOp,
    llvm::SmallVectorImpl<std::pair<mlir::Value, mlir::Value>> &ivPairs) {
  ivPairs.clear();
  if (!producerOp || !consumerOp)
    return false;
  llvm::SmallVector<mlir::affine::AffineForOp, 4> producerChain;
  llvm::SmallVector<mlir::affine::AffineForOp, 4> consumerChain;
  for (mlir::Operation *p = producerOp->getParentOp(); p; p = p->getParentOp())
    if (auto f = mlir::dyn_cast<mlir::affine::AffineForOp>(p))
      producerChain.push_back(f);
  for (mlir::Operation *p = consumerOp->getParentOp(); p; p = p->getParentOp())
    if (auto f = mlir::dyn_cast<mlir::affine::AffineForOp>(p))
      consumerChain.push_back(f);
  if (producerChain.empty() || consumerChain.empty())
    return false;
  // Pair innermost-first up to the smaller chain length. The producer
  // and consumer must agree on each paired loop's iteration domain.
  unsigned n = std::min(producerChain.size(), consumerChain.size());
  for (unsigned i = 0; i < n; ++i) {
    auto pf = producerChain[i];
    auto cf = consumerChain[i];
    if (!pf.hasConstantLowerBound() || !cf.hasConstantLowerBound() ||
        !pf.hasConstantUpperBound() || !cf.hasConstantUpperBound() ||
        pf.getConstantLowerBound() != cf.getConstantLowerBound() ||
        pf.getConstantUpperBound() != cf.getConstantUpperBound() ||
        pf.getStepAsInt() != cf.getStepAsInt())
      return false;
    ivPairs.push_back({pf.getInductionVar(), cf.getInductionVar()});
  }
  return true;
}

bool affineAccessEqualUnderIVSub(
    mlir::Operation *consumerLoad, mlir::Operation *producerStore,
    llvm::ArrayRef<std::pair<mlir::Value, mlir::Value>> ivPairs) {
  auto cLoad = mlir::dyn_cast<mlir::affine::AffineLoadOp>(consumerLoad);
  auto pStore = mlir::dyn_cast<mlir::affine::AffineStoreOp>(producerStore);
  if (!cLoad || !pStore)
    return false;
  if (cLoad.getAffineMap() != pStore.getAffineMap())
    return false;
  auto cOps = cLoad.getMapOperands();
  auto pOps = pStore.getMapOperands();
  if (cOps.size() != pOps.size())
    return false;
  llvm::DenseMap<mlir::Value, mlir::Value> sub;
  for (auto &kv : ivPairs)
    sub.try_emplace(kv.first, kv.second);
  for (auto [pIdx, cIdx] : llvm::zip(pOps, cOps)) {
    mlir::Value pResolved = pIdx;
    auto it = sub.find(pIdx);
    if (it != sub.end())
      pResolved = it->second;
    if (pResolved == cIdx)
      continue;
    auto fp = tryFoldAffineValueLocal(pResolved);
    auto fc = tryFoldAffineValueLocal(cIdx);
    if (fp && fc && *fp == *fc)
      continue;
    return false;
  }
  return true;
}

/// Local copy of tryFoldAffineValue (the version in DataRecomputation.cpp is
/// file-local). Folds arith.constant, statically trip-1 affine.for IVs, and
/// affine.apply chains to a constant.
static std::optional<int64_t> tryFoldAffineValueLocal(mlir::Value v) {
  if (auto cst = mlir::getConstantIntValue(v))
    return *cst;
  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(v)) {
    if (auto forOp = mlir::dyn_cast<mlir::affine::AffineForOp>(
            blockArg.getOwner()->getParentOp())) {
      if (blockArg == forOp.getInductionVar() &&
          forOp.hasConstantLowerBound() &&
          forOp.hasConstantUpperBound()) {
        int64_t lb = forOp.getConstantLowerBound();
        int64_t ub = forOp.getConstantUpperBound();
        int64_t step = forOp.getStepAsInt();
        if (step > 0 && ub > lb && (ub - lb) <= step)
          return lb;
      }
    }
    return std::nullopt;
  }
  if (auto applyOp = v.getDefiningOp<mlir::affine::AffineApplyOp>()) {
    llvm::SmallVector<mlir::Attribute, 4> attrs;
    auto idxTy = mlir::IndexType::get(applyOp.getContext());
    for (mlir::Value op : applyOp.getMapOperands()) {
      auto folded = tryFoldAffineValueLocal(op);
      if (!folded)
        return std::nullopt;
      attrs.push_back(mlir::IntegerAttr::get(idxTy, *folded));
    }
    llvm::SmallVector<mlir::Attribute, 1> results;
    if (mlir::failed(applyOp.getAffineMap().constantFold(attrs, results)))
      return std::nullopt;
    if (results.size() != 1)
      return std::nullopt;
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(results[0]))
      return intAttr.getInt();
  }
  return std::nullopt;
}

/// Return the induction variable of the innermost enclosing loop of op,
/// or a null Value when op is not inside any affine/scf loop.
static mlir::Value innermostEnclosingIV(mlir::Operation *op) {
  mlir::Operation *parent = op ? op->getParentOp() : nullptr;
  while (parent) {
    if (auto f = mlir::dyn_cast<mlir::affine::AffineForOp>(parent))
      return f.getInductionVar();
    if (auto f = mlir::dyn_cast<mlir::scf::ForOp>(parent))
      return f.getInductionVar();
    parent = parent->getParentOp();
  }
  return {};
}

/// Return the linear coefficient of dim `pos` in an affine expression.
/// Returns std::nullopt when the expression is not affine-linear in that
/// dim (e.g. uses mod/div/floordiv/ceildiv or multiplies the dim by a
/// non-constant).
static std::optional<int64_t> affineLinearCoef(mlir::AffineExpr expr,
                                               unsigned pos) {
  if (auto c = mlir::dyn_cast<mlir::AffineConstantExpr>(expr))
    return (int64_t)0;
  if (auto d = mlir::dyn_cast<mlir::AffineDimExpr>(expr))
    return (d.getPosition() == pos) ? (int64_t)1 : (int64_t)0;
  if (mlir::isa<mlir::AffineSymbolExpr>(expr))
    return (int64_t)0;
  auto bin = mlir::dyn_cast<mlir::AffineBinaryOpExpr>(expr);
  if (!bin)
    return std::nullopt;
  auto lhs = affineLinearCoef(bin.getLHS(), pos);
  auto rhs = affineLinearCoef(bin.getRHS(), pos);
  if (!lhs || !rhs)
    return std::nullopt;
  switch (bin.getKind()) {
  case mlir::AffineExprKind::Add:
    return *lhs + *rhs;
  case mlir::AffineExprKind::Mul: {
    if (auto c = mlir::dyn_cast<mlir::AffineConstantExpr>(bin.getLHS()))
      return c.getValue() * *rhs;
    if (auto c = mlir::dyn_cast<mlir::AffineConstantExpr>(bin.getRHS()))
      return *lhs * c.getValue();
    return std::nullopt;
  }
  default:
    return std::nullopt; // mod/floordiv/ceildiv
  }
}

/// Return true when value `v` transitively depends on `iv`.  Walks defining
/// ops; stops on block arguments (other than `iv` itself).
static bool dependsOnValue(mlir::Value v, mlir::Value iv) {
  if (!v || !iv)
    return false;
  llvm::SmallVector<mlir::Value, 8> worklist;
  llvm::SmallDenseSet<mlir::Value> visited;
  worklist.push_back(v);
  while (!worklist.empty()) {
    mlir::Value cur = worklist.pop_back_val();
    if (!visited.insert(cur).second)
      continue;
    if (cur == iv)
      return true;
    mlir::Operation *defOp = cur.getDefiningOp();
    if (!defOp)
      continue;
    for (mlir::Value o : defOp->getOperands())
      worklist.push_back(o);
  }
  return false;
}

/// Estimate the per-iteration access stride (in elements) of a load/store
/// access relative to the induction variable `iv`.  Returns:
///   - 0 when the access is invariant in iv (same address every iteration).
///   - a positive integer when the stride is statically determinable.
///   - std::nullopt when the stride cannot be determined; callers should
///     treat this as pessimistically full-cache-line.
/// Handles both memref.load/store and affine.load/store.
static std::optional<int64_t>
estimateAccessStrideElements(mlir::Operation *accessOp, mlir::Value iv) {
  if (!accessOp || !iv)
    return std::nullopt;

  mlir::MemRefType memrefTy;
  mlir::ValueRange rawIndices;
  mlir::AffineMap map;
  mlir::ValueRange mapOperands;

  if (auto op = mlir::dyn_cast<mlir::memref::LoadOp>(accessOp)) {
    memrefTy = op.getMemRefType();
    rawIndices = op.getIndices();
  } else if (auto op = mlir::dyn_cast<mlir::memref::StoreOp>(accessOp)) {
    memrefTy = op.getMemRefType();
    rawIndices = op.getIndices();
  } else if (auto op = mlir::dyn_cast<mlir::affine::AffineLoadOp>(accessOp)) {
    memrefTy = op.getMemRefType();
    map = op.getAffineMap();
    mapOperands = op.getMapOperands();
  } else if (auto op = mlir::dyn_cast<mlir::affine::AffineStoreOp>(accessOp)) {
    memrefTy = op.getMemRefType();
    map = op.getAffineMap();
    mapOperands = op.getMapOperands();
  } else {
    return std::nullopt;
  }

  unsigned rank = memrefTy.getRank();
  if (rank == 0)
    return (int64_t)0; // scalar memref: always same address

  llvm::ArrayRef<int64_t> shape = memrefTy.getShape();
  // Trailing-dim-product for each dim gives the element stride that a
  // coefficient of 1 in that dim contributes (row-major).
  llvm::SmallVector<int64_t, 4> trailing(rank, 1);
  for (int i = (int)rank - 2; i >= 0; --i) {
    if (shape[i + 1] < 0)
      return std::nullopt; // dynamic trailing dim
    trailing[i] = trailing[i + 1] * shape[i + 1];
  }

  int64_t totalStride = 0;

  if (map) {
    // affine.load/store: analyze each result expression.
    int ivPos = -1;
    for (unsigned i = 0; i < mapOperands.size(); ++i) {
      if (mapOperands[i] == iv) {
        ivPos = (int)i;
        break;
      }
    }
    if (ivPos < 0)
      return (int64_t)0; // iv not among the map operands

    // In MLIR, affine map operands are laid out as [dims..., symbols...].
    // AffineDimExpr positions reference the dim portion. If the iv is
    // passed as a symbol, treat as unknown.
    unsigned numDims = map.getNumDims();
    if ((unsigned)ivPos >= numDims)
      return std::nullopt;

    for (unsigned i = 0; i < rank; ++i) {
      auto coef = affineLinearCoef(map.getResult(i), (unsigned)ivPos);
      if (!coef)
        return std::nullopt;
      totalStride += *coef * trailing[i];
    }
  } else {
    // memref.load/store: inspect each index.
    for (unsigned i = 0; i < rank; ++i) {
      mlir::Value idx = rawIndices[i];
      if (idx == iv) {
        totalStride += trailing[i];
        continue;
      }
      if (!dependsOnValue(idx, iv))
        continue; // invariant in this dim
      // Simple pattern: idx = iv * c  or  iv * c + k  (linear integer
      // arithmetic via arith.muli/addi).  Walk and try to extract a
      // constant coefficient; bail out otherwise.
      std::function<std::optional<int64_t>(mlir::Value)> coefOf =
          [&](mlir::Value v) -> std::optional<int64_t> {
        if (v == iv)
          return (int64_t)1;
        if (!dependsOnValue(v, iv))
          return (int64_t)0;
        mlir::Operation *d = v.getDefiningOp();
        if (!d)
          return std::nullopt;
        if (mlir::isa<mlir::arith::AddIOp>(d)) {
          auto lhs = coefOf(d->getOperand(0));
          auto rhs = coefOf(d->getOperand(1));
          if (!lhs || !rhs)
            return std::nullopt;
          return *lhs + *rhs;
        }
        if (mlir::isa<mlir::arith::MulIOp>(d)) {
          auto getConst = [](mlir::Value x) -> std::optional<int64_t> {
            auto *dx = x.getDefiningOp();
            if (!dx)
              return std::nullopt;
            if (auto c = mlir::dyn_cast<mlir::arith::ConstantIndexOp>(dx))
              return (int64_t)c.value();
            if (auto c = mlir::dyn_cast<mlir::arith::ConstantIntOp>(dx))
              return (int64_t)c.value();
            return std::nullopt;
          };
          auto c0 = getConst(d->getOperand(0));
          auto c1 = getConst(d->getOperand(1));
          if (c0) {
            auto r = coefOf(d->getOperand(1));
            if (!r)
              return std::nullopt;
            return *c0 * *r;
          }
          if (c1) {
            auto l = coefOf(d->getOperand(0));
            if (!l)
              return std::nullopt;
            return *l * *c1;
          }
          return std::nullopt;
        }
        if (auto cast = mlir::dyn_cast<mlir::arith::IndexCastOp>(d))
          return coefOf(cast.getOperand());
        if (auto cast = mlir::dyn_cast<mlir::arith::IndexCastUIOp>(d))
          return coefOf(cast.getOperand());
        return std::nullopt;
      };
      auto c = coefOf(idx);
      if (!c)
        return std::nullopt;
      totalStride += *c * trailing[i];
    }
  }

  // Normalize: negative strides access in reverse but still touch one line
  // per iteration conservatively; use magnitude.
  if (totalStride < 0)
    totalStride = -totalStride;
  return totalStride;
}

/// Return the element size (bytes) of a load/store op's memref.
static unsigned accessElementBytes(mlir::Operation *accessOp) {
  mlir::Type elemTy;
  if (auto o = mlir::dyn_cast<mlir::memref::LoadOp>(accessOp))
    elemTy = o.getMemRefType().getElementType();
  else if (auto o = mlir::dyn_cast<mlir::memref::StoreOp>(accessOp))
    elemTy = o.getMemRefType().getElementType();
  else if (auto o = mlir::dyn_cast<mlir::affine::AffineLoadOp>(accessOp))
    elemTy = o.getMemRefType().getElementType();
  else if (auto o = mlir::dyn_cast<mlir::affine::AffineStoreOp>(accessOp))
    elemTy = o.getMemRefType().getElementType();
  if (!elemTy || !elemTy.isIntOrFloat())
    return 8; // conservative
  unsigned bits = elemTy.getIntOrFloatBitWidth();
  return bits > 0 ? bits / 8 : 8;
}

/// Stride-aware per-issuance cost for a load/store at its current (or
/// intended) site.  Scales the cold-miss latency by the fraction of a
/// cache line touched per iteration of the innermost enclosing loop at
/// `siteOp`:
///
///   effBytes = min(cacheLine, stride*elemSize + 1)
///   cost     = ceil(missLatency * effBytes / cacheLine)
///
/// The `+1` pessimizes unaligned strided access that may straddle a line
/// boundary.  When there is no enclosing loop, or the stride cannot be
/// determined, the full miss latency is charged.
unsigned estimateAccessLatency(mlir::Operation *accessOp,
                                      mlir::Operation *siteOp,
                                      int64_t bufferBytes,
                                      const CacheParams &cache) {
  unsigned missLat = estimateLoadLatency(bufferBytes, cache);
  unsigned cacheLine = cache.cacheLineSize ? cache.cacheLineSize : 64;

  mlir::Value iv = innermostEnclosingIV(siteOp ? siteOp : accessOp);
  if (!iv)
    return missLat; // one-shot: charge a full cold miss.

  auto strideOpt = estimateAccessStrideElements(accessOp, iv);
  if (!strideOpt)
    return missLat; // unknown stride → pessimize as full line per iter.

  unsigned elemBytes = accessElementBytes(accessOp);
  uint64_t strideBytes = (uint64_t)(*strideOpt) * elemBytes + 1;
  uint64_t effBytes = std::min<uint64_t>(cacheLine, strideBytes);
  uint64_t num = (uint64_t)missLat * effBytes;
  unsigned scaled = (unsigned)((num + cacheLine - 1) / cacheLine);
  return scaled ? scaled : 1; // at least 1 cycle of cost.
}

/// Estimate the per-issuance cost of the partial-leaf loads cloned during
/// partial rematerialization.  Each leaf is priced against its buffer-size
/// cache tier and scaled by stride-aware cache-line amortization relative
/// to the innermost enclosing loop of `insertionPoint` (where the clones
/// will execute).
unsigned
estimateLeafLoadsCost(llvm::ArrayRef<mlir::Operation *> partialLeaves,
                      mlir::Operation *insertionPoint,
                      const CacheParams &cache,
                      llvm::DenseMap<mlir::Value, mlir::Operation *> &allocRootFor) {
  unsigned total = 0;
  for (mlir::Operation *leaf : partialLeaves) {
    mlir::Value memref;
    if (auto l = mlir::dyn_cast<mlir::memref::LoadOp>(leaf))
      memref = l.getMemRef();
    else if (auto l = mlir::dyn_cast<mlir::affine::AffineLoadOp>(leaf))
      memref = l.getMemRef();
    if (!memref) {
      total += cache.memLatency;
      continue;
    }
    mlir::Operation *root = lookupAllocRoot(memref, allocRootFor);
    int64_t sizeBytes = cache.l2Size + 1; // conservative if unknown
    if (root) {
      if (auto bs = estimateBufferSizeBytes(root))
        sizeBytes = *bs;
    }
    total += estimateAccessLatency(leaf, insertionPoint, sizeBytes, cache);
  }
  return total;
}



/// Build a write summary per allocation root, covering every function in the
/// module. Stores are attributed to roots by tracing their memref through
/// view-like ops via collectBaseMemrefs.
RootWriteMap
buildRootWriteMap(mlir::ModuleOp moduleOp,
                  llvm::DenseMap<mlir::Value, mlir::Operation *> &allocRootFor) {
  RootWriteMap result;

  auto attribute = [&](mlir::Value memref, mlir::Operation *storeOp) {
    llvm::SmallVector<mlir::Value, 2> bases;
    collectBaseMemrefs(memref, allocRootFor, bases);
    llvm::SmallDenseSet<mlir::Operation *, 2> attributedRoots;
    for (mlir::Value base : bases) {
      auto it = allocRootFor.find(base);
      if (it == allocRootFor.end())
        continue;
      if (!attributedRoots.insert(it->second).second)
        continue;
      auto &summary = result[it->second];
      summary.writeCount++;
      summary.singleWriter = (summary.writeCount == 1) ? storeOp : nullptr;
    }
  };

  moduleOp.walk([&](mlir::Operation *op) {
    if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op))
      attribute(store.getMemRef(), op);
    else if (auto store = mlir::dyn_cast<mlir::affine::AffineStoreOp>(op))
      attribute(store.getMemRef(), op);
    else if (auto call = mlir::dyn_cast<mlir::CallOpInterface>(op)) {
      for (mlir::Value operand : call->getOperands()) {
        if (!mlir::isa<mlir::MemRefType, mlir::LLVM::LLVMPointerType>(
                operand.getType()))
          continue;
        llvm::SmallVector<mlir::Value, 2> bases;
        collectBaseMemrefs(operand, allocRootFor, bases);
        for (mlir::Value base : bases) {
          auto it = allocRootFor.find(base);
          if (it != allocRootFor.end())
            result[it->second].escapesToCall = true;
        }
      }
    }
  });

  return result;
}

/// Look up an allocation root for a load-like op's memref value. Traces
/// through view-like casts so subview/cast users resolve to their root.
mlir::Operation *lookupAllocRoot(
    mlir::Value memref,
    llvm::DenseMap<mlir::Value, mlir::Operation *> &allocRootFor) {
  auto it = allocRootFor.find(memref);
  if (it != allocRootFor.end())
    return it->second;
  llvm::SmallVector<mlir::Value, 2> bases;
  collectBaseMemrefs(memref, allocRootFor, bases);
  for (mlir::Value base : bases) {
    auto it2 = allocRootFor.find(base);
    if (it2 != allocRootFor.end())
      return it2->second;
  }
  return nullptr;
}

/// Validate that an SSA value used by a partial leaf load is available at
/// insertionPoint. Mirrors the block-arg/dominance checks in
/// isRematerializable.
static bool isValueAvailableAt(mlir::Value operand,
                               mlir::Operation *insertionPoint,
                               mlir::DominanceInfo &domInfo,
                               const mlir::IRMapping *argMapping) {
  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
    // If the mapping has an entry for this block arg (function entry arg
    // for interproc remat OR loop IV for IV-substituted partial remat),
    // accept and check the mapped value's availability.
    if (argMapping && argMapping->contains(blockArg))
      return domInfo.dominates(argMapping->lookup(blockArg), insertionPoint);
    if (argMapping) {
      // Interprocedural mode (argMapping populated but this arg not in
      // it): only function-entry args are otherwise admissible, but
      // the contains() check above already failed. Reject.
      return false;
    }
    return domInfo.dominates(operand, insertionPoint);
  }
  mlir::Operation *defOp = operand.getDefiningOp();
  if (!defOp)
    return false;
  // SSA scope check: the operand's enclosing region must enclose
  // insertionPoint, otherwise the value is dead outside its defining
  // region (e.g. defined in a sibling structured-op body). MLIR's
  // op-level properlyDominates is too permissive — it returns true for
  // ops in sibling regions whose parent ops are block-ordered, which
  // does not imply the SSA value is in scope at the use site.
  mlir::Region *defRegion = defOp->getParentRegion();
  mlir::Region *ipRegion = insertionPoint->getParentRegion();
  if (!defRegion || !ipRegion)
    return false;
  if (defRegion != ipRegion && !defRegion->isAncestor(ipRegion))
    return false;
  return domInfo.properlyDominates(defOp, insertionPoint);
}

/// Returns true if the writer's execution is guaranteed to have completed
/// by the time control reaches insertionPoint. MLIR's DominanceInfo is
/// conservative about ops inside loops/branches; we walk up the ancestor
/// chain until we find a common-region ancestor and check sequential
/// ordering there.
///
/// This is sound for partial rematerialization because the pass only
/// considers replacing loads that the original program actually executed
/// (SINGLE-provenance loads reaching insertionPoint) — if the writer is
/// statically reachable and we observe the downstream load firing, any
/// in-loop or in-branch wrapping of the writer must also have fired.
static bool writerReachesInsertionPoint(mlir::Operation *writer,
                                        mlir::Operation *insertionPoint) {
  // Find the nearest enclosing block common to both writer and the
  // insertion point, then compare the isBeforeInBlock order of the
  // ancestor of `writer` against the ancestor of `insertionPoint` at
  // that level.  This handles the case where the IP is in a deeper
  // nested region than the writer (e.g. writer in a sibling top-level
  // loop, IP inside a later loop body).
  for (mlir::Operation *ipAncestor = insertionPoint; ipAncestor;
       ipAncestor = ipAncestor->getParentOp()) {
    mlir::Block *ipBlock = ipAncestor->getBlock();
    if (!ipBlock)
      return false;
    mlir::Operation *wAncestor = writer;
    while (wAncestor && wAncestor->getBlock() != ipBlock)
      wAncestor = wAncestor->getParentOp();
    if (wAncestor)
      return wAncestor->isBeforeInBlock(ipAncestor);
  }
  return false;
}

/// Determine whether a non-chainable leaf load can be safely re-issued at
/// insertionPoint. A safe leaf's clone reads the same memref cell as the
/// original load did. See plan §3 for the write-once proxy rationale.
static bool isSafeLeafAt(
    mlir::Operation *loadOp, mlir::Operation *insertionPoint,
    mlir::DominanceInfo &domInfo,
    const mlir::IRMapping *argMapping,
    llvm::DenseMap<mlir::Value, mlir::Operation *> &allocRootFor,
    const RootWriteMap &rootWrites,
    PartialLeafReject &reject,
    bool skipOperandCheck = false) {
  reject = PartialLeafReject::None;

  mlir::Value memref;
  if (auto l = mlir::dyn_cast<mlir::memref::LoadOp>(loadOp))
    memref = l.getMemRef();
  else if (auto l = mlir::dyn_cast<mlir::affine::AffineLoadOp>(loadOp))
    memref = l.getMemRef();
  if (!memref)
    return false;

  mlir::Operation *root = lookupAllocRoot(memref, allocRootFor);
  if (!root) {
    reject = PartialLeafReject::NoAllocRoot;
    return false;
  }

  // Scoped alloca: the alloca's parent region must enclose insertionPoint.
  if (mlir::isa<mlir::memref::AllocaOp>(root)) {
    mlir::Region *allocRegion = root->getParentRegion();
    mlir::Region *ipRegion = insertionPoint->getParentRegion();
    if (!allocRegion || !ipRegion || !allocRegion->isAncestor(ipRegion)) {
      reject = PartialLeafReject::AllocaOutOfScope;
      return false;
    }
  }

  // Const globals (memref.global with constant attr, krnl.global) are
  // never written. Re-issuing a load against them is always safe with
  // respect to writers — skip the writer-summary checks. The operand
  // availability check at the end of this function still applies.
  bool isConstGlobal = false;
  if (auto g = mlir::dyn_cast<mlir::memref::GlobalOp>(root))
    isConstGlobal = g.getConstant();
  else if (root->getName().getStringRef() == "krnl.global")
    isConstGlobal = true;

  if (!isConstGlobal) {
    auto it = rootWrites.find(root);
    if (it == rootWrites.end()) {
      reject = PartialLeafReject::MultiWrite;
      return false;
    }
    const RootWriteSummary &summary = it->second;
    if (summary.escapesToCall) {
      reject = PartialLeafReject::Escapes;
      return false;
    }
    if (summary.writeCount != 1 || !summary.singleWriter) {
      reject = PartialLeafReject::MultiWrite;
      return false;
    }
    if (!writerReachesInsertionPoint(summary.singleWriter, insertionPoint)) {
      reject = PartialLeafReject::WriterDoesNotDominate;
      return false;
    }
    // S2-A: block-order reachability accepts a writer wrapped in a
    // conditional (scf.if / affine.if) because the conditional itself
    // sits before insertionPoint -- but the writer may never have fired
    // if the condition was false.  Reject if any ancestor between the
    // writer and the common-block level is a conditional region op.
    // Loops are still accepted; S1-A's join-lattice sentinel ensures a
    // may-zero loop yields LEAKED rather than SINGLE upstream.
    for (mlir::Operation *ipAnc = insertionPoint; ipAnc;
         ipAnc = ipAnc->getParentOp()) {
      mlir::Operation *wAnc = summary.singleWriter;
      while (wAnc && wAnc->getBlock() != ipAnc->getBlock())
        wAnc = wAnc->getParentOp();
      if (wAnc) {
        for (mlir::Operation *cur = summary.singleWriter->getParentOp();
             cur; cur = cur->getParentOp()) {
          if (mlir::isa<mlir::scf::IfOp, mlir::affine::AffineIfOp>(cur)) {
            reject = PartialLeafReject::WriterDoesNotDominate;
            return false;
          }
          if (cur == wAnc)
            break;
        }
        break;
      }
    }
  }
  (void)domInfo;

  // Every operand of the leaf load (including memref and indices/map operands)
  // must be available at insertionPoint. Skipped under IV-substituted
  // partial remat: the caller will recurse into index operands via the
  // worklist (cloning their producers) and verify the memref operand
  // separately, so this strict check would over-reject.
  if (!skipOperandCheck) {
    for (mlir::Value operand : loadOp->getOperands()) {
      if (!isValueAvailableAt(operand, insertionPoint, domInfo, argMapping)) {
        reject = PartialLeafReject::OperandNotLive;
        return false;
      }
    }
  }

  return true;
}

/// Checks whether the SSA operand tree rooted at rootVal can be cloned
/// (rematerialized) just before insertionPoint.
///
/// argMapping: optional mapping from callee entry block arguments to caller
/// values (for interprocedural use).
///
/// loadProv: if non-null, enables chaining through SINGLE-provenance loads.
/// When the operand tree encounters a memref.load whose provenance is SINGLE,
/// the load is substituted with the unique store's value and recursion
/// continues into that value's operand tree.
///
/// loadSubs: output pairs of (loadResult, storeValue) for each load that was
/// chained through. Used by rematerializeAt to wire up the substitutions.
///
/// On success, opsToClone is filled in topological (def-before-use) order.

bool isRematerializable(
    mlir::Value rootVal, mlir::Operation *insertionPoint,
    mlir::DominanceInfo &domInfo,
    llvm::SmallVectorImpl<mlir::Operation *> &opsToClone,
    const mlir::IRMapping *argMapping,
    const LoadProvenanceMap *loadProv,
    llvm::SmallVectorImpl<std::pair<mlir::Value, mlir::Value>> *loadSubs,
    PartialRematOpts *partial) {
  opsToClone.clear();
  llvm::SmallVector<mlir::Value, 8> worklist;
  llvm::SmallDenseSet<mlir::Value> visited;
  llvm::SmallDenseSet<mlir::Operation *> added;
  unsigned chainDepth = 0;
  worklist.push_back(rootVal);

  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;

    // Block argument: check dominance (possibly through argMapping).
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(current)) {
      if (argMapping && argMapping->contains(blockArg)) {
        // Mapping has an entry: function-entry arg (interproc) or loop
        // IV (IV-substituted partial remat). Check the mapped value
        // dominates insertionPoint.
        mlir::Value mapped = argMapping->lookup(blockArg);
        if (!domInfo.dominates(mapped, insertionPoint))
          return false;
      } else if (argMapping) {
        // argMapping is non-null but this arg isn't in it. Reject —
        // the caller intended to substitute, and an unmapped arg is
        // not in scope at insertionPoint by construction.
        return false;
      } else {
        if (!domInfo.dominates(current, insertionPoint))
          return false;
      }
      continue;
    }

    mlir::Operation *defOp = current.getDefiningOp();
    if (!defOp)
      return false;

    // Special case: handle memref.load / affine.load.
    if (mlir::isa<mlir::memref::LoadOp, mlir::affine::AffineLoadOp>(defOp)) {
      // Partial mode: prefer cloning as a partial leaf (no recursion into
      // the load's upstream store value). This lets the pass rematerialize
      // expressions whose chain would otherwise dead-end at a non-chainable
      // inner load — Strategy 2 already tried the full chain path.
      if (partial && partial->allow && partial->leaves &&
          partial->rootWrites && partial->allocRootFor) {
        if (partial->leaves->size() >= partial->maxLeaves)
          return false;
        PartialLeafReject why = PartialLeafReject::None;
        if (isSafeLeafAt(defOp, insertionPoint, domInfo, argMapping,
                         *partial->allocRootFor, *partial->rootWrites, why,
                         partial->ivSubstitution)) {
          if (added.insert(defOp).second)
            opsToClone.push_back(defOp);
          partial->leaves->push_back(defOp);
          // IV-substituted mode: also recurse into the load's INDEX
          // operands so the index-computing SSA chain is cloned at the
          // consumer site. The memref operand was verified by
          // isSafeLeafAt's availability check (or is the leaf root).
          if (partial->ivSubstitution) {
            mlir::Value memref;
            mlir::ValueRange indices;
            if (auto l = mlir::dyn_cast<mlir::memref::LoadOp>(defOp)) {
              memref = l.getMemRef();
              indices = l.getIndices();
            } else if (auto l =
                           mlir::dyn_cast<mlir::affine::AffineLoadOp>(defOp)) {
              memref = l.getMemRef();
              indices = l.getMapOperands();
            }
            // Memref must be available at insertionPoint without cloning
            // (allocs / globals live at function/module scope and should
            // dominate). Reject if not.
            if (memref && !isValueAvailableAt(memref, insertionPoint, domInfo,
                                               argMapping)) {
              partial->lastReject = PartialLeafReject::OperandNotLive;
              return false;
            }
            for (mlir::Value idx : indices)
              worklist.push_back(idx);
          }
          continue;
        }
        partial->lastReject = why;
        // Fall through to chain path as a last resort.
      }

      // Strict-chain path: chain through SINGLE-provenance loads only.
      if (loadProv && loadSubs && chainDepth < kMaxChainDepth) {
        auto provIt = loadProv->find(defOp);
        if (provIt != loadProv->end()) {
          const auto &provSet = provIt->second;
          if (provSet.size() == 1 && !provSet.contains(nullptr)) {
            auto *uniqueStore = *provSet.begin();
            mlir::Value chainedVal = getStoredValue(uniqueStore);
            if (chainedVal) {
              loadSubs->push_back({defOp->getResult(0), chainedVal});
              worklist.push_back(chainedVal);
              ++chainDepth;
              continue;
            }
          }
        }
      }
      // Not SINGLE, not resolvable, or depth exceeded.
      return false;
    }

    // Reject ops with memory effects or calls.
    if (!mlir::isMemoryEffectFree(defOp))
      return false;
    if (mlir::isa<mlir::CallOpInterface>(defOp))
      return false;

    // Leaf op (no operands): e.g., constants.
    if (defOp->getNumOperands() == 0) {
      if (added.insert(defOp).second)
        opsToClone.push_back(defOp);
      continue;
    }

    // Interior op: add and recurse into operands.
    if (added.insert(defOp).second) {
      opsToClone.push_back(defOp);
      if (opsToClone.size() > kMaxRematOps)
        return false;
    }
    for (mlir::Value operand : defOp->getOperands())
      worklist.push_back(operand);
  }

  // Reverse for topological order (defs before uses).
  std::reverse(opsToClone.begin(), opsToClone.end());
  return true;
}

/// Clone the ops in opsToClone at insertionPoint and return the rematerialized
/// value corresponding to originalVal.
///
/// loadSubs: pairs of (loadResult, storeValue) from isRematerializable.
/// Each load result is mapped to the (possibly cloned) store value so that
/// downstream cloned ops see the substituted value instead of the load.
mlir::Value rematerializeAt(
    mlir::Value originalVal, mlir::Operation *insertionPoint,
    llvm::ArrayRef<mlir::Operation *> opsToClone,
    const mlir::IRMapping *argMapping,
    llvm::ArrayRef<std::pair<mlir::Value, mlir::Value>> loadSubs) {
  mlir::OpBuilder builder(insertionPoint);
  mlir::IRMapping mapping;
  if (argMapping) {
    // Copy callee-arg → caller-value mappings.
    for (auto &[from, to] :
         llvm::make_range(argMapping->getValueMap().begin(),
                          argMapping->getValueMap().end()))
      mapping.map(from, to);
  }
  for (mlir::Operation *op : opsToClone) {
    // Eagerly resolve load substitutions whose store value is now available.
    for (auto &[loadResult, storeValue] : loadSubs) {
      if (!mapping.contains(loadResult)) {
        mlir::Value resolved = mapping.lookupOrDefault(storeValue);
        if (resolved != storeValue || !storeValue.getDefiningOp()) {
          // storeValue was cloned or is a dominating value — resolve now.
          mapping.map(loadResult, resolved);
        }
      }
    }
    builder.clone(*op, mapping);
  }
  // Final pass: resolve any loadSubs for dominating values not in opsToClone.
  for (auto &[loadResult, storeValue] : loadSubs) {
    if (!mapping.contains(loadResult))
      mapping.map(loadResult, mapping.lookupOrDefault(storeValue));
  }
  return mapping.lookupOrDefault(originalVal);
}

} // namespace dr
