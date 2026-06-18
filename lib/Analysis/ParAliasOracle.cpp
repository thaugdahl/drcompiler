//===- ParAliasOracle.cpp - Intra-module conflict oracle (M0) -----------===//

#include "drcompiler/Analysis/ParAliasOracle.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace drcompiler {
namespace par {

Value ParAliasOracle::allocationRoot(Value memref) {
  // Follow the view chain (memref.subview / cast / view / reinterpret_cast /
  // expand_shape / collapse_shape all implement ViewLikeOpInterface).
  while (Operation *def = memref.getDefiningOp()) {
    auto view = dyn_cast<ViewLikeOpInterface>(def);
    if (!view)
      break;
    memref = view.getViewSource();
  }
  return memref;
}

/// Resolve a func.call's callee and check it transitively touches no memory.
/// `visiting` guards against recursion (a cycle is treated as impure).
static bool isPureFuncImpl(func::FuncOp fn,
                           llvm::DenseSet<Operation *> &visiting) {
  if (fn.isExternal())
    return false;
  if (!visiting.insert(fn.getOperation()).second)
    return false; // recursion: conservative
  bool pure = true;
  fn.walk([&](Operation *op) {
    if (op == fn.getOperation())
      return; // walk is root-inclusive; the func.func op itself is not an effect
    if (auto call = dyn_cast<func::CallOp>(op)) {
      auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr());
      if (!callee || !isPureFuncImpl(callee, visiting))
        pure = false;
      return;
    }
    if (op->hasTrait<OpTrait::IsTerminator>())
      return;
    if (!isMemoryEffectFree(op))
      pure = false;
  });
  visiting.erase(fn.getOperation());
  return pure;
}

bool ParAliasOracle::isPureCall(Operation *op) {
  auto call = dyn_cast<func::CallOp>(op);
  if (!call)
    return false;
  auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      call, call.getCalleeAttr());
  if (!callee)
    return false;
  llvm::DenseSet<Operation *> visiting;
  return isPureFuncImpl(callee, visiting);
}

ConflictKind ParAliasOracle::classify(Operation *a, Operation *b,
                                      unsigned loopDepth) const {
  bool aWrites = isa<affine::AffineWriteOpInterface>(a);
  bool bWrites = isa<affine::AffineWriteOpInterface>(b);
  // Two reads never conflict.
  if (!aWrites && !bWrites)
    return ConflictKind::None;

  affine::MemRefAccess sa(a), da(b);

  // Tier 1 — same SSA memref, affine accesses: the exact polyhedral test.
  if (sa.memref == da.memref) {
    affine::DependenceResult result = affine::checkMemrefAccessDependence(
        sa, da, loopDepth, /*dependenceConstraints=*/nullptr,
        /*dependenceComponents=*/nullptr);
    return affine::hasDependence(result) ? ConflictKind::Carried
                                         : ConflictKind::None;
  }

  // Distinct memref SSA values.  Adopt the affine dialect's aliasing model --
  // the same assumption checkMemrefAccessDependence / affine::isLoopParallel /
  // affine-loop-fusion already rely on: memrefs with DISTINCT allocation roots
  // (distinct allocs, or distinct function arguments) do not alias.  Two views
  // of the SAME root (e.g. subviews of one buffer) may overlap at an offset, so
  // stay conservative -- stricter than raw affine, which would wrongly treat
  // them as independent.
  Value ra = allocationRoot(sa.memref);
  Value rb = allocationRoot(da.memref);
  if (ra == rb)
    return ConflictKind::Unknown; // same underlying buffer via views
  return ConflictKind::None;      // distinct roots = distinct memory
}

ConflictKind ParAliasOracle::axisConflict(Operation *loopOp) const {
  auto loop = dyn_cast<affine::AffineForOp>(loopOp);
  if (!loop)
    return ConflictKind::Unknown; // M0: SCF dependence not modeled.

  // Gather affine accesses; flag any opaque effectful op (a call, a non-affine
  // memref.store, …) we cannot reason about ⇒ conservative.
  SmallVector<Operation *, 16> accesses;
  bool sawOpaque = false;
  loop->walk([&](Operation *op) {
    if (op == loopOp)
      return;
    if (isa<affine::AffineReadOpInterface, affine::AffineWriteOpInterface>(op)) {
      accesses.push_back(op);
      return;
    }
    if (op->hasTrait<OpTrait::IsTerminator>())
      return;
    if (isa<affine::AffineForOp, affine::AffineIfOp, affine::AffineParallelOp>(
            op))
      return;
    // A call to a provably-pure function touches no memory — benign under
    // parallel execution (M4 interprocedural consumption, sound subset).
    if (!isMemoryEffectFree(op) && !isPureCall(op))
      sawOpaque = true;
  });

  unsigned loopDepth = affine::getNestingDepth(loopOp) + 1;
  ConflictKind acc = ConflictKind::None;
  for (Operation *a : accesses) {
    for (Operation *b : accesses) {
      acc = worst(acc, classify(a, b, loopDepth));
      if (acc == ConflictKind::Unknown)
        return acc; // already maximal
    }
  }
  if (sawOpaque)
    acc = worst(acc, ConflictKind::Unknown);
  return acc;
}

} // namespace par
} // namespace drcompiler
