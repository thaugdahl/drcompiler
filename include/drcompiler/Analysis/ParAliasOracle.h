//===- ParAliasOracle.h - Intra-module conflict oracle (M0) -------------===//
//
// The single legality predicate the bubble-widening front-end calls.  Given two
// memory accesses and the loop whose distribution is being tested, it reports
// the strongest conflict between them along that axis.  Tiered:
//
//   Tier 0  provenance disjointness  — distinct allocation roots never alias.
//   Tier 1  affine exact             — same memref, affine accesses: the
//                                       polyhedral dependence test.
//   Tier 2  conservative             — anything else is treated as a conflict.
//
// M0 scope: Tiers 0–2 over affine.for nests; cross-procedure forwarding (the
// EnrichedCallGraph path) and SCF dependence are deferred to later milestones
// (see PARALLEL_BUBBLE_SPEC.md §2, §5).
//
//===----------------------------------------------------------------------===//

#ifndef DRCOMPILER_ANALYSIS_PARALIASORACLE_H
#define DRCOMPILER_ANALYSIS_PARALIASORACLE_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace drcompiler {
namespace par {

/// The strongest relationship between two accesses along one loop axis,
/// ordered so that `max` yields the worst (least parallelizable) verdict.
enum class ConflictKind {
  None = 0,            ///< provably independent along this axis
  LoopIndependent = 1, ///< a same-iteration dependence (privatizable)
  Carried = 2,         ///< a loop-carried dependence: distributing is illegal
  Unknown = 3,         ///< cannot prove independence ⇒ treated as a conflict
};

inline ConflictKind worst(ConflictKind a, ConflictKind b) {
  return static_cast<int>(a) >= static_cast<int>(b) ? a : b;
}

class ParAliasOracle {
public:
  /// Classify the relationship between accesses `a` and `b` (affine or generic
  /// load/store ops) along the loop nest depth `loopDepth` (1-based, as
  /// `checkMemrefAccessDependence` expects).
  ConflictKind classify(mlir::Operation *a, mlir::Operation *b,
                        unsigned loopDepth) const;

  /// Worst conflict among all access pairs in `loopOp`'s body when `loopOp` is
  /// treated as a distributed (parallel) axis.  `ConflictKind::None` ⇒ the axis
  /// is parallel.  Returns `Unknown` for non-affine loops in M0.
  ConflictKind axisConflict(mlir::Operation *loopOp) const;

  /// Trace a memref SSA value back through view-like ops (subview/cast/view/…)
  /// to its allocation root.  Two distinct alloc/alloca roots never alias.
  static mlir::Value allocationRoot(mlir::Value memref);

  /// True when `op` is a call to a provably *pure* function — its callee is
  /// defined (non-external) and transitively touches no memory (operates only
  /// on SSA arguments).  Such a call is race-free to run per parallel
  /// iteration.  Conservative: external / unresolved / recursive callees and
  /// any memory effect ⇒ false.  (M4 interprocedural call consumption: the
  /// sound subset; per-iteration-disjoint impure calls via forwarding analysis
  /// are a later step.)
  static bool isPureCall(mlir::Operation *op);
};

} // namespace par
} // namespace drcompiler

#endif // DRCOMPILER_ANALYSIS_PARALIASORACLE_H
