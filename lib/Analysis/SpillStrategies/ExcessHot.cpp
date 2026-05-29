//===- ExcessHot.cpp - max(0, peak - budget) * spill_reload * trip ---------===//
//
// Charges spill cycles only at the hottest program point.  Cheapest of the
// three strategies; accurate when pressure has a clear peak (e.g. a single
// fused inner-loop body).
//
//===----------------------------------------------------------------------===//

#include "../PressureTrace.h"

namespace drcompiler {

uint64_t aggregateExcessHot(const PressureTrace &trace,
                            const RegisterParams &params,
                            const ArchHandler &arch,
                            const ArchParams &archParams,
                            uint64_t tripCount) {
  uint64_t total = 0;
  for (RegClass cls : allRegClasses()) {
    unsigned peak = trace.peakLive[static_cast<unsigned>(cls)];
    unsigned budget = params.budgetFor(cls);
    if (peak <= budget)
      continue;
    unsigned excess = peak - budget;
    unsigned perSpill =
        arch.tuneSpillCost(params.spillReloadCycles, cls, archParams);
    total += uint64_t(excess) * perSpill * std::max<uint64_t>(tripCount, 1);
  }
  return total;
}

} // namespace drcompiler
