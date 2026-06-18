//===- ParAliasOracle.cpp - Intra-module conflict oracle (M0) -----------===//

#include "drcompiler/Analysis/ParAliasOracle.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

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

/// True when `v` is the result of an allocation op (a fresh, named buffer).
static bool isAllocLike(Value v) {
  Operation *def = v.getDefiningOp();
  return def && isa<memref::AllocOp, memref::AllocaOp>(def);
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

  // Tier 0 — distinct allocation roots never alias.
  Value ra = allocationRoot(sa.memref);
  Value rb = allocationRoot(da.memref);
  if (ra != rb && isAllocLike(ra) && isAllocLike(rb))
    return ConflictKind::None;

  // Tier 2 — different/unprovable memrefs (function args, globals, non-affine):
  // sound-conservative.  Refined by cross-procedure forwarding in M4.
  return ConflictKind::Unknown;
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
    if (!isMemoryEffectFree(op))
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
