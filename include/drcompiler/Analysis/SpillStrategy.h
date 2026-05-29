//===- SpillStrategy.h - Pluggable spill cost aggregation ------------------===//
//
// A SpillStrategy turns per-program-point live counts (plus the per-class
// register budget) into a single scalar spill-cycle estimate.  Three
// strategies ship in v1:
//
//   ExcessHot   : max(0, peak_live - budget) * spill_reload * trip_count
//                 (cheap: looks only at the hot point.  Best when the
//                  region's pressure profile has a clear peak.)
//
//   SumExcess   : sum over program points of max(0, live - budget),
//                 scaled by spill_reload (no trip multiplier, because
//                 the per-point sum already represents intra-region
//                 spill traffic).
//                 (more faithful when pressure is spread.)
//
//   GraphColor  : Chaitin-Briggs-style interference-graph coloring with
//                 spill counting per class.  More accurate but O(n^2)
//                 in distinct live values.
//
//===----------------------------------------------------------------------===//

#ifndef DRCOMPILER_ANALYSIS_SPILLSTRATEGY_H
#define DRCOMPILER_ANALYSIS_SPILLSTRATEGY_H

#include "llvm/ADT/StringRef.h"

namespace drcompiler {

enum class SpillStrategy {
  ExcessHot,
  SumExcess,
  GraphColor,
};

/// Parse a strategy name from a JSON/CLI string.  Returns ExcessHot for
/// unknown / empty inputs.
SpillStrategy parseSpillStrategy(llvm::StringRef name);

/// Canonical name (matches JSON `arch.spill_strategy`).
llvm::StringRef spillStrategyName(SpillStrategy s);

} // namespace drcompiler

#endif // DRCOMPILER_ANALYSIS_SPILLSTRATEGY_H
