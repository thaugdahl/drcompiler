//===- SumExcess.cpp - sum over program points of max(0, live - budget) ----===//
//
// Walks every point in the trace and charges per-class excess at that point.
// More faithful than ExcessHot when pressure rises and falls across the
// region (e.g. interleaved producers and consumers).  No trip multiplier:
// the per-point sum already captures intra-region spill traffic.
//
//===----------------------------------------------------------------------===//

#include "../PressureTrace.h"

namespace drcompiler {

uint64_t aggregateSumExcess(const PressureTrace &trace,
                            const RegisterParams &params,
                            const ArchHandler &arch,
                            const ArchParams &archParams) {
  uint64_t total = 0;
  for (const PressurePoint &pt : trace.points) {
    for (RegClass cls : allRegClasses()) {
      unsigned live = pt.perClassLive[static_cast<unsigned>(cls)];
      unsigned budget = params.budgetFor(cls);
      if (live <= budget)
        continue;
      unsigned excess = live - budget;
      unsigned perSpill =
          arch.tuneSpillCost(params.spillReloadCycles, cls, archParams);
      total += uint64_t(excess) * perSpill;
    }
  }
  return total;
}

} // namespace drcompiler
