//===- CacheCostModel.cpp - Cache-aware cost model -----------------------===//
//
//===----------------------------------------------------------------------===//

#include "drcompiler/Transforms/DataRecomputation/CacheCostModel.h"

#include "drcompiler/Analysis/MachineModel.h"
#include "drcompiler/Transforms/Utils/OpDispatchUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <functional>

namespace dr {

namespace {

/// An SSA value is a compute "leaf" (cost 0, depth 0) when it is a load: the
/// value is already resident in a register or memory and is not recomputed.
bool isComputeLeaf(mlir::Operation *defOp) {
  return mlir::isa<mlir::memref::LoadOp, mlir::affine::AffineLoadOp,
                   mlir::LLVM::LoadOp>(defOp);
}

/// Single memoized walk over the recompute cone of `val`. Returns the longest
/// weighted dependency path ("critical path") to `val`, and accumulates the
/// total weighted op cost into `totalCost` (each op counted once). Loads and
/// block arguments are inputs (depth 0). The def-use operand graph is acyclic
/// for non-block-arg values, so memoized recursion terminates.
unsigned computeCostWalk(mlir::Value val,
                         const drcompiler::CpuCostModel &costModel,
                         llvm::DenseMap<mlir::Value, unsigned> &depthMemo,
                         llvm::SmallDenseSet<mlir::Operation *> &countedOps,
                         unsigned &totalCost) {
  if (auto it = depthMemo.find(val); it != depthMemo.end())
    return it->second;

  unsigned depth = 0;
  if (!mlir::isa<mlir::BlockArgument>(val)) {
    if (mlir::Operation *defOp = val.getDefiningOp()) {
      if (!isComputeLeaf(defOp)) {
        unsigned operandDepth = 0;
        for (mlir::Value operand : defOp->getOperands())
          operandDepth = std::max(
              operandDepth, computeCostWalk(operand, costModel, depthMemo,
                                            countedOps, totalCost));
        unsigned c = costModel.opCost(defOp);
        depth = operandDepth + c;
        if (countedOps.insert(defOp).second)
          totalCost += c;
      }
    }
  }

  depthMemo[val] = depth;
  return depth;
}

} // namespace

unsigned estimateComputeCost(mlir::Value val,
                             const drcompiler::CpuCostModel &costModel,
                             unsigned issueWidth) {
  llvm::DenseMap<mlir::Value, unsigned> depthMemo;
  llvm::SmallDenseSet<mlir::Operation *> countedOps;
  unsigned totalCost = 0;
  unsigned criticalPath =
      computeCostWalk(val, costModel, depthMemo, countedOps, totalCost);

  // The realized cost of recomputing an expression on an out-of-order,
  // superscalar core is bounded below by (a) its dependency critical path —
  // you cannot finish before the longest dependent chain — and (b) its
  // throughput, i.e. total work spread across the target's issue ports. The
  // previous model returned `totalCost` (sum of all ops), which over-priced
  // wide/independent expressions and biased every keep-vs-recompute decision
  // toward buffering. For a purely linear dependent chain criticalPath ==
  // totalCost, so this is a no-op there and changes only wide expressions.
  // See COSTMODEL_FINDINGS_CLAUDE.md §2.4 (A1).
  //
  // The width is the target's, from ArchParams::issueWidth (handler default,
  // JSON-overridable via `arch.issue_width`) — it was a hardcoded 4 until the
  // portability cleanup. Clamped to >= 1: a malformed `issue_width: 0` must not
  // divide by zero, and 1 is the honest reading of "no superscalar overlap".
  unsigned width = std::max(issueWidth, 1u);
  unsigned throughput = (totalCost + width - 1) / width;
  return std::max(criticalPath, throughput);
}

std::optional<int64_t> estimateBufferSizeBytes(mlir::Operation *allocOp) {
  mlir::MemRefType memrefTy;

  if (auto alloc = mlir::dyn_cast<mlir::memref::AllocOp>(allocOp))
    memrefTy = alloc.getType();
  else if (auto alloca = mlir::dyn_cast<mlir::memref::AllocaOp>(allocOp))
    memrefTy = alloca.getType();
  else if (auto g = mlir::dyn_cast<mlir::memref::GlobalOp>(allocOp))
    memrefTy = g.getType();
  else if (allocOp && allocOp->getNumResults() == 1 &&
           mlir::isa<mlir::MemRefType>(allocOp->getResult(0).getType()) &&
           (allocOp->getName().getStringRef() == "krnl.global"))
    // krnl.global is unregistered but its single memref result carries the
    // static type we need.
    memrefTy = mlir::cast<mlir::MemRefType>(allocOp->getResult(0).getType());
  else
    return std::nullopt;

  if (!memrefTy.hasStaticShape())
    return std::nullopt;

  int64_t numElements = 1;
  for (int64_t dim : memrefTy.getShape())
    numElements *= dim;

  // Element may be int, float, or a vector of int/float.  Calling
  // `getElementTypeBitWidth()` directly asserts on vector elements, so
  // dispatch by hand.
  mlir::Type elemTy = memrefTy.getElementType();
  unsigned elementBits = 0;
  if (auto vt = mlir::dyn_cast<mlir::VectorType>(elemTy)) {
    if (!vt.getElementType().isIntOrFloat())
      return std::nullopt;
    elementBits =
        vt.getElementType().getIntOrFloatBitWidth() * vt.getNumElements();
  } else if (elemTy.isIntOrFloat()) {
    elementBits = elemTy.getIntOrFloatBitWidth();
  } else {
    return std::nullopt;
  }
  if (elementBits == 0)
    return std::nullopt;

  return numElements * (elementBits / 8);
}

unsigned estimateLoadLatency(int64_t bufferSizeBytes,
                             const CacheParams &cache) {
  // Private levels (L1/L2) are not shared, so a co-tenant cannot evict them --
  // but a working set only stays L2-resident if it leaves room for the other
  // co-resident arrays (and the SMT sibling / prefetcher); judge L2 residency
  // against l2OccupancyPct of the size (100 = full, the default for DR).
  if (bufferSizeBytes <= (int64_t)cache.l1Size)
    return cache.l1Latency;
  unsigned occ = cache.l2OccupancyPct ? cache.l2OccupancyPct : 100;
  if (bufferSizeBytes <= (int64_t)cache.l2Size * (int64_t)occ / 100)
    return cache.l2Latency;
  // The shared LLC is contended: a reuse only counts on the fraction we are
  // guaranteed, l3Size / llcSharers. Beyond that, assume evicted (memLatency).
  int64_t effectiveL3 =
      drcompiler::MachineModel::effectiveLLC(cache.l3Size, cache.llcSharers);
  if (cache.l3Size > 0 && bufferSizeBytes <= effectiveL3)
    return cache.l3Latency;
  return cache.memLatency;
}

std::optional<int64_t> traceToConstant(mlir::Value val,
                                       const EnrichedCallGraph &callGraph) {
  if (mlir::Operation *defOp = val.getDefiningOp()) {
    if (auto cIdx = mlir::dyn_cast<mlir::arith::ConstantIndexOp>(defOp))
      return cIdx.value();
    if (auto cInt = mlir::dyn_cast<mlir::arith::ConstantIntOp>(defOp))
      return cInt.value();
    if (auto cast = mlir::dyn_cast<mlir::arith::IndexCastOp>(defOp))
      return traceToConstant(cast.getIn(), callGraph);
    return std::nullopt;
  }

  auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(val);
  if (!blockArg)
    return std::nullopt;

  auto *parentOp = blockArg.getOwner()->getParentOp();
  auto funcOp = mlir::dyn_cast_or_null<mlir::FunctionOpInterface>(parentOp);
  if (!funcOp)
    return std::nullopt;

  unsigned argIdx = blockArg.getArgNumber();
  auto cgIt = callGraph.find(funcOp.getOperation());
  if (cgIt == callGraph.end() || cgIt->second.empty())
    return std::nullopt;

  std::optional<int64_t> commonVal;
  for (const auto &edge : cgIt->second) {
    if (argIdx >= edge.callSiteOp->getNumOperands())
      return std::nullopt;
    mlir::Value callerOperand = edge.callSiteOp->getOperand(argIdx);
    mlir::Operation *callerDef = callerOperand.getDefiningOp();
    std::optional<int64_t> traced;
    if (callerDef) {
      if (auto c = mlir::dyn_cast<mlir::arith::ConstantIndexOp>(callerDef))
        traced = c.value();
      else if (auto c = mlir::dyn_cast<mlir::arith::ConstantIntOp>(callerDef))
        traced = c.value();
      else if (auto cast = mlir::dyn_cast<mlir::arith::IndexCastOp>(callerDef))
        if (auto *inner = cast.getIn().getDefiningOp()) {
          if (auto c = mlir::dyn_cast<mlir::arith::ConstantIndexOp>(inner))
            traced = c.value();
          else if (auto c = mlir::dyn_cast<mlir::arith::ConstantIntOp>(inner))
            traced = c.value();
        }
    }
    if (!traced)
      return std::nullopt;
    if (commonVal && *commonVal != *traced)
      return std::nullopt;
    commonVal = traced;
  }
  return commonVal;
}

std::optional<int64_t> estimateTripCount(mlir::Operation *loopOp,
                                         const EnrichedCallGraph &callGraph) {
  if (auto affFor = mlir::dyn_cast<mlir::affine::AffineForOp>(loopOp)) {
    int64_t step = affFor.getStepAsInt();
    if (step <= 0) return std::nullopt;
    if (affFor.hasConstantLowerBound() && affFor.hasConstantUpperBound()) {
      int64_t lb = affFor.getConstantLowerBound();
      int64_t ub = affFor.getConstantUpperBound();
      if (ub <= lb) return 0;
      return (ub - lb + step - 1) / step;
    }
    if (affFor.hasConstantLowerBound()) {
      auto ubOperands = affFor.getUpperBoundOperands();
      if (ubOperands.size() == 1) {
        auto ubVal = traceToConstant(ubOperands[0], callGraph);
        if (ubVal) {
          int64_t lb = affFor.getConstantLowerBound();
          if (*ubVal <= lb) return 0;
          return (*ubVal - lb + step - 1) / step;
        }
      }
    }
    return std::nullopt;
  }

  if (auto scfFor = mlir::dyn_cast<mlir::scf::ForOp>(loopOp)) {
    auto lbVal = traceToConstant(scfFor.getLowerBound(), callGraph);
    auto ubVal = traceToConstant(scfFor.getUpperBound(), callGraph);
    auto stepVal = traceToConstant(scfFor.getStep(), callGraph);
    if (lbVal && ubVal && stepVal && *stepVal > 0) {
      if (*ubVal <= *lbVal) return 0;
      return (*ubVal - *lbVal + *stepVal - 1) / *stepVal;
    }
    return std::nullopt;
  }

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Stride-aware access primitives.
//
// Analyze a load/store's per-iteration address stride relative to an induction
// variable. Shared by the spatial-locality footprint refinement below
// (estimateOpFootprintBytes) and by the partial-remat leaf cost in
// RematKernel.cpp (estimateAccessLatency).
//===----------------------------------------------------------------------===//

mlir::Value innermostEnclosingIV(mlir::Operation *op) {
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

unsigned accessElementBytes(mlir::Operation *accessOp) {
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

namespace {

/// Return the linear coefficient of dim `pos` in an affine expression.
/// Returns std::nullopt when the expression is not affine-linear in that
/// dim (e.g. uses mod/div/floordiv/ceildiv or multiplies the dim by a
/// non-constant).
std::optional<int64_t> affineLinearCoef(mlir::AffineExpr expr, unsigned pos) {
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
bool dependsOnValue(mlir::Value v, mlir::Value iv) {
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

} // namespace

std::optional<int64_t>
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

namespace {

/// Upper bound on the number of DISTINCT bytes a loop can touch in total: the
/// sum of static sizes of the distinct memrefs it (transitively) accesses. Used
/// to cap the footprint = bodyFP * tripCount product, which otherwise counts a
/// small array re-read every iteration as bytes * tripCount — a 64 B array read
/// 1000 times scored as 64000 B, with no temporal-reuse or spatial-locality
/// modeling. A loop cannot touch more distinct bytes than the arrays it
/// accesses contain. Returns nullopt when the loop contains a call or a
/// dynamically shaped / non-int-float access, where no static distinct-bytes
/// bound is available — callers then fall back to the uncapped product and so
/// never under-count. See COSTMODEL_FINDINGS_CLAUDE.md §2.4 (A2).
std::optional<int64_t> distinctMemrefBytes(mlir::Operation *loopOp) {
  llvm::SmallDenseSet<mlir::Value> seen;
  int64_t total = 0;
  bool uncappable = false;
  loopOp->walk([&](mlir::Operation *op) {
    // A call may touch arbitrary memory we cannot statically bound.
    if (mlir::isa<mlir::CallOpInterface>(op)) {
      uncappable = true;
      return mlir::WalkResult::interrupt();
    }
    if (!drcompiler::isAnyLoadOp(op) && !drcompiler::isAnyStoreOp(op))
      return mlir::WalkResult::advance();
    mlir::Value memref = drcompiler::getLoadStoreMemref(op);
    if (!memref)
      return mlir::WalkResult::advance();
    if (!seen.insert(memref).second)
      return mlir::WalkResult::advance();
    auto memrefTy = mlir::dyn_cast<mlir::MemRefType>(memref.getType());
    if (!memrefTy || !memrefTy.hasStaticShape()) {
      uncappable = true;
      return mlir::WalkResult::interrupt();
    }
    mlir::Type elemTy = memrefTy.getElementType();
    if (!elemTy.isIntOrFloat()) {
      uncappable = true;
      return mlir::WalkResult::interrupt();
    }
    int64_t numElems = 1;
    for (int64_t dim : memrefTy.getShape())
      numElems *= dim;
    total += numElems * (int64_t)(elemTy.getIntOrFloatBitWidth() / 8);
    return mlir::WalkResult::advance();
  });
  if (uncappable || total < 0) // total<0 = multiply overflow; do not bound
    return std::nullopt;
  return total;
}

/// bodyFP * trip, capped by the distinct bytes the loop can touch (temporal
/// reuse). Falls back to the raw product when no static bound is available, so
/// this can only move an over-estimate toward (never below) the truth.
int64_t cappedLoopFootprint(mlir::Operation *loopOp, int64_t bodyFP,
                            int64_t trip) {
  int64_t looped = bodyFP * trip;
  if (looped < 0) // multiply overflow — leave uncapped
    return looped;
  if (std::optional<int64_t> distinct = distinctMemrefBytes(loopOp))
    return std::min<int64_t>(looped, *distinct);
  return looped;
}

int64_t estimateOpFootprintBytes(mlir::Operation *op,
                                 const CacheParams &cache) {
  if (drcompiler::isAnyLoadOp(op) || drcompiler::isAnyStoreOp(op)) {
    mlir::Type elemTy = drcompiler::getLoadStoreElementType(op);
    unsigned elemBytes = (elemTy && elemTy.isIntOrFloat())
                             ? elemTy.getIntOrFloatBitWidth() / 8
                             : 8;
    // Spatial-locality refinement. The per-iteration footprint of a scalar
    // access is NOT one element when it skips cache lines: a strided / gather
    // access touches (and so can evict) up to a full line every iteration,
    // while a contiguous (stride-1) stream amortizes one line over
    // line/elem iterations and so contributes one element per iteration.
    // When the per-iteration stride relative to the innermost enclosing loop
    // is statically known we charge min(line, stride*elem); otherwise we fall
    // back to the per-element estimate. The refinement is monotone-upward
    // (stride<=1 and unknown-stride are unchanged from the old per-element
    // value); only provably-strided accesses grow, fixing the prior
    // under-count that priced a column sweep like a contiguous stream.
    if (mlir::Value iv = innermostEnclosingIV(op)) {
      if (std::optional<int64_t> stride = estimateAccessStrideElements(op, iv)) {
        if (*stride > 1) {
          int64_t line = cache.cacheLineSize ? cache.cacheLineSize : 64;
          return std::min<int64_t>(line, *stride * (int64_t)elemBytes);
        }
      }
    }
    return elemBytes;
  }

  for (mlir::Value result : op->getResults()) {
    if (auto vecTy = mlir::dyn_cast<mlir::VectorType>(result.getType())) {
      if (mlir::isa<mlir::vector::LoadOp, mlir::vector::TransferReadOp>(op))
        return (vecTy.getNumElements() * vecTy.getElementTypeBitWidth()) / 8;
    }
  }
  if (mlir::isa<mlir::vector::StoreOp, mlir::vector::TransferWriteOp>(op)) {
    mlir::Value vecVal = op->getOperand(0);
    if (auto vecTy = mlir::dyn_cast<mlir::VectorType>(vecVal.getType()))
      return (vecTy.getNumElements() * vecTy.getElementTypeBitWidth()) / 8;
  }

  if (auto callOp = mlir::dyn_cast<mlir::CallOpInterface>(op)) {
    int64_t total = 0;
    for (mlir::Value operand : callOp->getOperands()) {
      auto memrefTy = mlir::dyn_cast<mlir::MemRefType>(operand.getType());
      if (!memrefTy)
        continue;
      if (memrefTy.hasStaticShape()) {
        int64_t numElems = 1;
        for (int64_t dim : memrefTy.getShape())
          numElems *= dim;
        unsigned elemBits = memrefTy.getElementTypeBitWidth();
        total += elemBits > 0 ? numElems * (elemBits / 8) : 0;
      } else {
        total += cache.l2Size;
      }
    }
    return total;
  }

  return 0;
}

int64_t estimateBlockFootprintBytes(mlir::Block &block,
                                    const CacheParams &cache,
                                    const EnrichedCallGraph &callGraph);

int64_t estimateBlockFootprintBytes(mlir::Block &block,
                                    const CacheParams &cache,
                                    const EnrichedCallGraph &callGraph) {
  int64_t total = 0;

  for (mlir::Operation &op : block) {
    if (mlir::isa<mlir::affine::AffineForOp, mlir::scf::ForOp>(&op)) {
      mlir::Region &bodyRegion = op.getRegion(0);
      if (!bodyRegion.empty()) {
        int64_t bodyFP =
            estimateBlockFootprintBytes(bodyRegion.front(), cache, callGraph);
        auto tc = estimateTripCount(&op, callGraph);
        total += cappedLoopFootprint(&op, bodyFP,
                                     tc ? *tc : kDefaultTripCount);
      }
      continue;
    }

    if (mlir::isa<mlir::scf::IfOp, mlir::affine::AffineIfOp>(&op)) {
      int64_t thenFP = 0, elseFP = 0;
      mlir::Region &thenRegion = op.getRegion(0);
      if (!thenRegion.empty())
        thenFP = estimateBlockFootprintBytes(thenRegion.front(), cache,
                                             callGraph);
      if (op.getNumRegions() > 1) {
        mlir::Region &elseRegion = op.getRegion(1);
        if (!elseRegion.empty())
          elseFP = estimateBlockFootprintBytes(elseRegion.front(), cache,
                                               callGraph);
      }
      total += std::max(thenFP, elseFP);
      continue;
    }

    if (mlir::isa<mlir::scf::WhileOp>(&op)) {
      if (op.getNumRegions() > 1) {
        mlir::Region &bodyRegion = op.getRegion(1);
        if (!bodyRegion.empty()) {
          int64_t bodyFP =
              estimateBlockFootprintBytes(bodyRegion.front(), cache, callGraph);
          total += cappedLoopFootprint(&op, bodyFP, kDefaultTripCount);
        }
      }
      continue;
    }

    total += estimateOpFootprintBytes(&op, cache);
  }

  return total;
}

mlir::Operation *findAncestorInBlock(mlir::Operation *op,
                                     mlir::Block *targetBlock) {
  mlir::Operation *current = op;
  while (current) {
    if (current->getBlock() == targetBlock)
      return current;
    current = current->getParentOp();
  }
  return nullptr;
}

int64_t sumFootprintBetween(mlir::Operation *from, mlir::Operation *to,
                            const CacheParams &cache,
                            const EnrichedCallGraph &callGraph) {
  int64_t total = 0;
  for (mlir::Operation *it = from->getNextNode(); it && it != to;
       it = it->getNextNode()) {
    if (it->getNumRegions() > 0) {
      if (mlir::isa<mlir::affine::AffineForOp, mlir::scf::ForOp>(it)) {
        mlir::Region &body = it->getRegion(0);
        if (!body.empty()) {
          int64_t bodyFP =
              estimateBlockFootprintBytes(body.front(), cache, callGraph);
          auto tc = estimateTripCount(it, callGraph);
          total += cappedLoopFootprint(it, bodyFP,
                                       tc ? *tc : kDefaultTripCount);
        }
        continue;
      }
      if (mlir::isa<mlir::scf::IfOp, mlir::affine::AffineIfOp>(it)) {
        int64_t thenFP = 0, elseFP = 0;
        if (!it->getRegion(0).empty())
          thenFP = estimateBlockFootprintBytes(it->getRegion(0).front(), cache,
                                               callGraph);
        if (it->getNumRegions() > 1 && !it->getRegion(1).empty())
          elseFP = estimateBlockFootprintBytes(it->getRegion(1).front(), cache,
                                               callGraph);
        total += std::max(thenFP, elseFP);
        continue;
      }
      if (mlir::isa<mlir::scf::WhileOp>(it)) {
        if (it->getNumRegions() > 1 && !it->getRegion(1).empty()) {
          int64_t bodyFP = estimateBlockFootprintBytes(
              it->getRegion(1).front(), cache, callGraph);
          total += cappedLoopFootprint(it, bodyFP, kDefaultTripCount);
        }
        continue;
      }
      for (mlir::Region &region : it->getRegions())
        if (!region.empty())
          total +=
              estimateBlockFootprintBytes(region.front(), cache, callGraph);
      continue;
    }
    total += estimateOpFootprintBytes(it, cache);
  }
  return total;
}

int64_t sumFootprintAfter(mlir::Operation *op, const CacheParams &cache,
                          const EnrichedCallGraph &callGraph) {
  mlir::Block *block = op->getBlock();
  if (!block)
    return 0;
  return sumFootprintBetween(op, block->getTerminator(), cache, callGraph);
}

int64_t sumFootprintBefore(mlir::Operation *op,
                           const CacheParams &cache,
                           const EnrichedCallGraph &callGraph) {
  mlir::Block *block = op->getBlock();
  if (!block || block->empty())
    return 0;
  int64_t total = 0;
  for (mlir::Operation &it : *block) {
    if (&it == op)
      break;
    if (it.getNumRegions() > 0) {
      if (mlir::isa<mlir::affine::AffineForOp, mlir::scf::ForOp>(&it)) {
        mlir::Region &body = it.getRegion(0);
        if (!body.empty()) {
          int64_t bodyFP =
              estimateBlockFootprintBytes(body.front(), cache, callGraph);
          auto tc = estimateTripCount(&it, callGraph);
          total += cappedLoopFootprint(&it, bodyFP,
                                       tc ? *tc : kDefaultTripCount);
        }
        continue;
      }
      for (mlir::Region &region : it.getRegions())
        if (!region.empty())
          total +=
              estimateBlockFootprintBytes(region.front(), cache, callGraph);
      continue;
    }
    total += estimateOpFootprintBytes(&it, cache);
  }
  return total;
}

bool memrefAccessedInOp(mlir::Operation *op, mlir::Value memref) {
  bool found = false;
  op->walk([&](mlir::Operation *inner) {
    if (found) return mlir::WalkResult::interrupt();
    if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(inner)) {
      if (load.getMemRef() == memref) { found = true; return mlir::WalkResult::interrupt(); }
    } else if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(inner)) {
      if (store.getMemRef() == memref) { found = true; return mlir::WalkResult::interrupt(); }
    } else if (auto load = mlir::dyn_cast<mlir::affine::AffineLoadOp>(inner)) {
      if (load.getMemRef() == memref) { found = true; return mlir::WalkResult::interrupt(); }
    } else if (auto store = mlir::dyn_cast<mlir::affine::AffineStoreOp>(inner)) {
      if (store.getMemRef() == memref) { found = true; return mlir::WalkResult::interrupt(); }
    }
    return mlir::WalkResult::advance();
  });
  return found;
}

bool memrefAccessedInRange(mlir::Operation *from, mlir::Operation *to,
                           mlir::Value memref) {
  for (mlir::Operation *it = from->getNextNode(); it && it != to;
       it = it->getNextNode()) {
    if (memrefAccessedInOp(it, memref))
      return true;
  }
  return false;
}

} // namespace

int64_t estimateInterveningFootprint(mlir::Operation *storeOp,
                                     mlir::Operation *loadOp,
                                     const CacheParams &cache,
                                     const EnrichedCallGraph &callGraph) {
  auto storeFn = storeOp->getParentOfType<mlir::FunctionOpInterface>();
  auto loadFn = loadOp->getParentOfType<mlir::FunctionOpInterface>();
  if (storeFn != loadFn)
    return cache.l2Size;

  mlir::Block *storeBlock = storeOp->getBlock();
  mlir::Block *loadBlock = loadOp->getBlock();

  if (storeBlock == loadBlock)
    return sumFootprintBetween(storeOp, loadOp, cache, callGraph);

  if (mlir::Operation *storeAnc =
          findAncestorInBlock(storeOp, loadBlock)) {
    int64_t fp = sumFootprintAfter(storeOp, cache, callGraph);
    mlir::Operation *cur = storeOp->getParentOp();
    while (cur && cur != storeAnc) {
      mlir::Block *curBlock = cur->getBlock();
      if (curBlock)
        fp += sumFootprintAfter(cur, cache, callGraph);
      cur = cur->getParentOp();
    }
    fp += sumFootprintBetween(storeAnc, loadOp, cache, callGraph);
    return fp;
  }

  if (mlir::Operation *loadAnc =
          findAncestorInBlock(loadOp, storeBlock)) {
    int64_t fp = sumFootprintBetween(storeOp, loadAnc, cache, callGraph);
    fp += sumFootprintBefore(loadOp, cache, callGraph);
    mlir::Operation *cur = loadOp->getParentOp();
    while (cur && cur != loadAnc) {
      mlir::Block *curBlock = cur->getBlock();
      if (curBlock)
        fp += sumFootprintBefore(cur, cache, callGraph);
      cur = cur->getParentOp();
    }
    return fp;
  }

  for (mlir::Operation *sp = storeOp; sp; sp = sp->getParentOp()) {
    mlir::Block *spBlock = sp->getBlock();
    if (!spBlock)
      continue;
    mlir::Operation *loadAnc = findAncestorInBlock(loadOp, spBlock);
    if (!loadAnc)
      continue;
    mlir::Operation *storeAnc = sp;
    int64_t fp = sumFootprintAfter(storeOp, cache, callGraph);
    mlir::Operation *cur = storeOp->getParentOp();
    while (cur && cur != storeAnc) {
      mlir::Block *curBlock = cur->getBlock();
      if (curBlock)
        fp += sumFootprintAfter(cur, cache, callGraph);
      cur = cur->getParentOp();
    }
    fp += sumFootprintBetween(storeAnc, loadAnc, cache, callGraph);
    cur = loadOp->getParentOp();
    while (cur && cur != loadAnc) {
      mlir::Block *curBlock = cur->getBlock();
      if (curBlock)
        fp += sumFootprintBefore(cur, cache, callGraph);
      cur = cur->getParentOp();
    }
    fp += sumFootprintBefore(loadOp, cache, callGraph);
    return fp;
  }

  return cache.l2Size;
}

void collectOperandMemrefs(mlir::Value val,
                           llvm::SmallDenseSet<mlir::Value> &memrefs) {
  llvm::SmallVector<mlir::Value, 8> worklist;
  llvm::SmallDenseSet<mlir::Value> visited;
  worklist.push_back(val);

  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    if (mlir::isa<mlir::BlockArgument>(current))
      continue;
    mlir::Operation *defOp = current.getDefiningOp();
    if (!defOp)
      continue;

    if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(defOp)) {
      memrefs.insert(load.getMemRef());
      continue;
    }
    if (auto load = mlir::dyn_cast<mlir::affine::AffineLoadOp>(defOp)) {
      memrefs.insert(load.getMemRef());
      continue;
    }
    if (mlir::isa<mlir::LLVM::LoadOp>(defOp))
      continue;

    for (mlir::Value operand : defOp->getOperands())
      worklist.push_back(operand);
  }
}

unsigned estimateOperandReloadPenalty(mlir::Value storedVal,
                                      mlir::Operation *storeOp,
                                      mlir::Operation *loadOp,
                                      int64_t storeToLoadFootprint,
                                      const CacheParams &cache) {
  llvm::SmallDenseSet<mlir::Value> operandMemrefs;
  collectOperandMemrefs(storedVal, operandMemrefs);

  if (operandMemrefs.empty())
    return 0;

  auto storeFn = storeOp->getParentOfType<mlir::FunctionOpInterface>();
  auto loadFn = loadOp->getParentOfType<mlir::FunctionOpInterface>();

  unsigned totalPenalty = 0;
  unsigned coldCount = 0;

  for (mlir::Value memref : operandMemrefs) {
    bool warm = false;

    if (storeFn == loadFn && storeOp->getBlock() == loadOp->getBlock()) {
      warm = memrefAccessedInRange(storeOp, loadOp, memref);
    } else if (storeFn == loadFn) {
      mlir::Block *loadBlock = loadOp->getBlock();
      if (loadBlock && !loadBlock->empty()) {
        for (mlir::Operation &it : *loadBlock) {
          if (&it == loadOp) break;
          if (memrefAccessedInOp(&it, memref)) {
            warm = true;
            break;
          }
        }
      }
    }

    if (!warm)
      coldCount++;
  }

  if (coldCount == 0)
    return 0;

  if (storeToLoadFootprint > (int64_t)cache.l1Size) {
    unsigned evictedLatency = estimateLoadLatency(storeToLoadFootprint, cache);
    unsigned delta = evictedLatency > cache.l1Latency
                         ? evictedLatency - cache.l1Latency
                         : 0;
    totalPenalty = (delta * coldCount) / operandMemrefs.size();
  }

  return totalPenalty;
}

MaterializationDecision
decideBufferStrategy(const MaterializationInputs &inputs,
                     const CacheParams &cache,
                     const drcompiler::ArchHandler &arch,
                     const drcompiler::ArchParams &archParams) {
  unsigned effectiveLoadLatency = inputs.loadLatency;
  if (inputs.storeToLoadFootprint > 0) {
    int64_t workingSet = inputs.bufferSizeBytes + inputs.storeToLoadFootprint;
    effectiveLoadLatency = estimateLoadLatency(workingSet, cache);
  }

  unsigned memKeep = inputs.numConsumers * effectiveLoadLatency;
  unsigned aluKeep = inputs.aluCost + 1; // store once
  unsigned regKeep = inputs.regCyclesKeep;

  unsigned memRecompute =
      inputs.numConsumers * (inputs.leafLoadCost + inputs.operandPenalty);
  unsigned aluRecompute = inputs.numConsumers * inputs.aluCost;
  unsigned regRecompute = inputs.regCyclesRecompute;

  unsigned totalKeep =
      arch.combineCosts(memKeep, regKeep, aluKeep, archParams);
  unsigned totalRecompute = arch.combineCosts(memRecompute, regRecompute,
                                              aluRecompute, archParams);

  MaterializationDecision d;
  d.recompute = totalRecompute <= totalKeep;
  d.aluCost = inputs.aluCost;
  d.leafLoadCost = inputs.leafLoadCost;
  d.loadLatency = effectiveLoadLatency;
  d.numConsumers = inputs.numConsumers;
  d.bufferSizeBytes = inputs.bufferSizeBytes;
  d.storeToLoadFootprint = inputs.storeToLoadFootprint;
  d.operandPenalty = inputs.operandPenalty;
  d.memCyclesKeep = memKeep;
  d.memCyclesRecompute = memRecompute;
  d.regCyclesKeep = regKeep;
  d.regCyclesRecompute = regRecompute;
  d.aluCyclesKeep = aluKeep;
  d.aluCyclesRecompute = aluRecompute;
  d.totalKeep = totalKeep;
  d.totalRecompute = totalRecompute;
  return d;
}

BufferElimCostDecision decideBufferElimination(const BufferElimCostInputs &i) {
  unsigned keep = i.numLoads * i.loadLatency
                + i.numStores * i.storeLatency
                + i.allocOverheadCycles
                + i.capacityPenaltyCycles;

  // Effective number of computes the program will actually run, assuming
  // optimal CSE within each parent function. Falls back to numLoads when
  // the caller did not provide a distinct-group count.
  unsigned effectiveComputes =
      (i.numDistinctComputes > 0) ? i.numDistinctComputes : i.numLoads;
  unsigned compute = effectiveComputes * i.perElemComputeCost;

  unsigned elim = compute + i.codeBloatPenalty + i.regPressurePenalty;

  return BufferElimCostDecision{/*eliminate=*/elim <= keep, keep, elim};
}

} // namespace dr
