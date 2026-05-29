//===- RegisterPressureAnalysis.h - Per-region register pressure -----------===//
//
// AnalysisManager-managed analysis.  The constructor caches an
// `mlir::Liveness` over the anchor operation; queries against any region
// inside that anchor reuse the cached liveness.
//
// The analysis is *configurable* at query time: register budgets and the
// spill strategy are supplied per `query()`, so a single pass can ask
// "what if the budget were X?" or "what would SumExcess say?" without
// reconstructing liveness.
//
//===----------------------------------------------------------------------===//

#ifndef DRCOMPILER_ANALYSIS_REGISTERPRESSUREANALYSIS_H
#define DRCOMPILER_ANALYSIS_REGISTERPRESSUREANALYSIS_H

#include "drcompiler/Analysis/ArchHandler.h"
#include "drcompiler/Analysis/RegisterClass.h"
#include "drcompiler/Analysis/SpillStrategy.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"

#include <array>
#include <cstdint>

namespace drcompiler {

/// Per-class snapshot indexed by RegClass enum.
using RegClassCounts = std::array<unsigned, kNumRegClasses>;

/// Sum of two RegClassCounts; helper for aggregation paths.
inline RegClassCounts addCounts(RegClassCounts a, RegClassCounts b) {
  RegClassCounts out{};
  for (unsigned i = 0; i < kNumRegClasses; ++i)
    out[i] = a[i] + b[i];
  return out;
}

/// Result of a pressure query against a region.
struct PressureResult {
  /// Per-class peak live count across the region.
  RegClassCounts peakLive{};

  /// Per-class total excess (live > budget) summed across program points;
  /// units: register-points, not cycles.  Multiplied by spill_reload (and
  /// possibly trip count) inside the strategy to get cycles.
  RegClassCounts totalExcess{};

  /// Total estimated spill cycles, after the chosen SpillStrategy has run.
  /// This is the headline number the cost model consumes.
  uint64_t totalSpillCycles = 0;

  /// Per-op pressure snapshot.  Populated when `recordPerOp` is set in
  /// the query options.  Useful for the diagnostic pass and for
  /// `analyzeHypothetical`.
  llvm::DenseMap<mlir::Operation *, RegClassCounts> liveAtOp;
};

/// Query-level options.
struct PressureQuery {
  RegisterParams params;
  SpillStrategy strategy = SpillStrategy::ExcessHot;
  /// Trip count for ExcessHot scaling.  1 = treat the region as straight-line.
  uint64_t tripCount = 1;
  /// When true, populate `PressureResult::liveAtOp`.  Costs O(ops * peak).
  bool recordPerOp = false;
};

/// AnalysisManager-managed.  The same class is reused at both FuncOp and
/// ModuleOp anchor scopes (per the dual-scope decision); pick the scope by
/// the pass's anchor op type.
class RegisterPressureAnalysis {
public:
  /// AM ctor: caches a Liveness over `anchor`.  All queries against
  /// regions inside `anchor` will reuse this cache.
  explicit RegisterPressureAnalysis(mlir::Operation *anchor);

  mlir::Operation *anchor() const { return anchorOp; }
  const mlir::Liveness &liveness() const { return livenessInfo; }

  /// Pressure summary for a region within the cached anchor.
  PressureResult query(mlir::Region &region,
                       const ArchHandler &arch,
                       const ArchParams &archParams,
                       const PressureQuery &opts) const;

  /// Pressure *if* `clonedOps` (with their full operand closure) were
  /// inserted at the end of `region`.  The original IR is not mutated;
  /// the cloned ops are interpreted symbolically.
  PressureResult queryHypothetical(
      mlir::Region &region,
      llvm::ArrayRef<mlir::Operation *> clonedOps,
      const ArchHandler &arch,
      const ArchParams &archParams,
      const PressureQuery &opts) const;

  /// Free-function form for callers that don't want AM caching.  Builds a
  /// fresh Liveness over the region's parent op.
  static PressureResult analyzeRegionStatic(mlir::Region &region,
                                            const ArchHandler &arch,
                                            const ArchParams &archParams,
                                            const PressureQuery &opts);

  /// Static-form hypothetical query (mirrors `queryHypothetical`).
  static PressureResult analyzeHypotheticalStatic(
      mlir::Region &region,
      llvm::ArrayRef<mlir::Operation *> clonedOps,
      const ArchHandler &arch,
      const ArchParams &archParams,
      const PressureQuery &opts);

private:
  mlir::Operation *anchorOp;
  mlir::Liveness livenessInfo;
};

} // namespace drcompiler

#endif // DRCOMPILER_ANALYSIS_REGISTERPRESSUREANALYSIS_H
