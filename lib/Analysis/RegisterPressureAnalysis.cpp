//===- RegisterPressureAnalysis.cpp - Per-region register pressure ---------===//

#include "drcompiler/Analysis/RegisterPressureAnalysis.h"

#include "PressureTrace.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>

namespace drcompiler {

namespace {

/// Classify a value's type for the given arch.  Memoised inside the trace.
RegClassRequirement classifyAndCache(mlir::Value v, const ArchHandler &arch,
                                     const ArchParams &archParams,
                                     PressureTrace &trace) {
  auto it = trace.classification.find(v);
  if (it != trace.classification.end())
    return it->second;
  RegClassRequirement req = arch.classifyType(v.getType(), archParams);
  trace.classification[v] = req;
  return req;
}

/// Add a value to a per-class count snapshot using its classification.
void addValueToCounts(RegClassCounts &counts, RegClassRequirement req) {
  counts[static_cast<unsigned>(req.cls)] += req.count;
}

void updatePeak(RegClassCounts &peak, const RegClassCounts &cur) {
  for (unsigned i = 0; i < kNumRegClasses; ++i)
    peak[i] = std::max(peak[i], cur[i]);
}

/// Collect a stable, deterministic snapshot of values currently live across
/// `op` according to the liveness analysis.  The set order is determined
/// by program-order of the block plus the op's operand list, so the result
/// is reproducible across runs.
void collectLiveSet(mlir::Operation *op, const mlir::Liveness &liveness,
                    llvm::SmallVectorImpl<mlir::Value> &out) {
  mlir::Block *block = op->getBlock();
  if (!block)
    return;
  const mlir::LivenessBlockInfo *bi = liveness.getLiveness(block);
  if (!bi)
    return;
  auto raw = bi->currentlyLiveValues(op);
  out.assign(raw.begin(), raw.end());
}

/// Walk `region` in program order and populate trace points.  The same
/// liveness analysis is reused (it was constructed over an enclosing op).
void buildTrace(mlir::Region &region, const mlir::Liveness &liveness,
                const ArchHandler &arch, const ArchParams &archParams,
                PressureTrace &trace) {
  // Walk every op nested inside the region; this gives intra-region pressure
  // including bodies of inner loops/blocks (which is what callers want for a
  // loop-body region).  `walk` visits each op exactly once.
  region.walk([&](mlir::Operation *op) {
    PressurePoint pt;
    pt.op = op;
    collectLiveSet(op, liveness, pt.liveSet);
    for (mlir::Value v : pt.liveSet) {
      RegClassRequirement req = classifyAndCache(v, arch, archParams, trace);
      addValueToCounts(pt.perClassLive, req);
    }
    updatePeak(trace.peakLive, pt.perClassLive);
    trace.points.push_back(std::move(pt));
  });
}

/// Apply the chosen SpillStrategy to a built trace.
uint64_t aggregateSpills(SpillStrategy strategy, const PressureTrace &trace,
                         const RegisterParams &params,
                         const ArchHandler &arch,
                         const ArchParams &archParams, uint64_t tripCount) {
  switch (strategy) {
  case SpillStrategy::ExcessHot:
    return aggregateExcessHot(trace, params, arch, archParams, tripCount);
  case SpillStrategy::SumExcess:
    return aggregateSumExcess(trace, params, arch, archParams);
  case SpillStrategy::GraphColor:
    return aggregateGraphColor(trace, params, arch, archParams);
  }
  return 0;
}

/// Populate the trace-wide totals into a PressureResult.
PressureResult materialiseResult(const PressureTrace &trace,
                                 uint64_t spillCycles,
                                 const RegisterParams &params,
                                 const PressureQuery &opts) {
  PressureResult out;
  out.peakLive = trace.peakLive;
  out.totalSpillCycles = spillCycles;
  // totalExcess: sum across program points of max(0, live - budget).
  // Provided regardless of strategy because the per-class delta is useful
  // diagnostic data.
  for (const PressurePoint &pt : trace.points) {
    for (RegClass cls : allRegClasses()) {
      unsigned live = pt.perClassLive[static_cast<unsigned>(cls)];
      unsigned budget = params.budgetFor(cls);
      if (live > budget)
        out.totalExcess[static_cast<unsigned>(cls)] += live - budget;
    }
  }
  if (opts.recordPerOp) {
    out.liveAtOp.reserve(trace.points.size());
    for (const PressurePoint &pt : trace.points)
      out.liveAtOp[pt.op] = pt.perClassLive;
  }
  return out;
}

/// Compute the transitive operand closure of `seeds` restricted to ops not
/// in their own def-use cycle.  Used by `queryHypothetical` to model the
/// extra values that materialising the clones would keep live.
void collectOperandClosure(llvm::ArrayRef<mlir::Operation *> seeds,
                           llvm::SmallVectorImpl<mlir::Value> &outValues,
                           llvm::DenseSet<mlir::Operation *> &visited) {
  llvm::SmallVector<mlir::Operation *, 16> worklist(seeds.begin(),
                                                     seeds.end());
  while (!worklist.empty()) {
    mlir::Operation *op = worklist.pop_back_val();
    if (!visited.insert(op).second)
      continue;
    for (mlir::Value res : op->getResults())
      outValues.push_back(res);
    for (mlir::Value v : op->getOperands()) {
      if (auto *def = v.getDefiningOp()) {
        if (!visited.contains(def))
          worklist.push_back(def);
      } else {
        outValues.push_back(v);
      }
    }
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// RegisterPressureAnalysis
//===----------------------------------------------------------------------===//

RegisterPressureAnalysis::RegisterPressureAnalysis(mlir::Operation *anchor)
    : anchorOp(anchor), livenessInfo(anchor) {}

PressureResult
RegisterPressureAnalysis::query(mlir::Region &region, const ArchHandler &arch,
                                const ArchParams &archParams,
                                const PressureQuery &opts) const {
  PressureTrace trace;
  buildTrace(region, livenessInfo, arch, archParams, trace);
  uint64_t spillCycles = aggregateSpills(opts.strategy, trace, opts.params,
                                          arch, archParams, opts.tripCount);
  return materialiseResult(trace, spillCycles, opts.params, opts);
}

PressureResult RegisterPressureAnalysis::queryHypothetical(
    mlir::Region &region, llvm::ArrayRef<mlir::Operation *> clonedOps,
    const ArchHandler &arch, const ArchParams &archParams,
    const PressureQuery &opts) const {
  // Build base trace.
  PressureTrace trace;
  buildTrace(region, livenessInfo, arch, archParams, trace);

  // Augment with the def-use closure of `clonedOps`.  All values in the
  // closure are conservatively added to *every* point's live set: this is
  // worst-case (assumes the clones would live across the entire region).
  // Callers wanting finer placement can split the region first.
  llvm::SmallVector<mlir::Value, 32> extraValues;
  llvm::DenseSet<mlir::Operation *> visited;
  collectOperandClosure(clonedOps, extraValues, visited);

  RegClassCounts extraCounts{};
  for (mlir::Value v : extraValues) {
    RegClassRequirement req = classifyAndCache(v, arch, archParams, trace);
    addValueToCounts(extraCounts, req);
  }
  // Add the extra counts to every point and update the peak.
  for (PressurePoint &pt : trace.points) {
    for (unsigned i = 0; i < kNumRegClasses; ++i)
      pt.perClassLive[i] += extraCounts[i];
    updatePeak(trace.peakLive, pt.perClassLive);
    for (mlir::Value v : extraValues)
      pt.liveSet.push_back(v);
  }

  uint64_t spillCycles = aggregateSpills(opts.strategy, trace, opts.params,
                                          arch, archParams, opts.tripCount);
  return materialiseResult(trace, spillCycles, opts.params, opts);
}

PressureResult RegisterPressureAnalysis::analyzeRegionStatic(
    mlir::Region &region, const ArchHandler &arch,
    const ArchParams &archParams, const PressureQuery &opts) {
  // Anchor liveness at the smallest enclosing op that owns this region.
  mlir::Operation *anchor = region.getParentOp();
  mlir::Liveness localLiveness(anchor);

  PressureTrace trace;
  buildTrace(region, localLiveness, arch, archParams, trace);
  uint64_t spillCycles = aggregateSpills(opts.strategy, trace, opts.params,
                                          arch, archParams, opts.tripCount);
  return materialiseResult(trace, spillCycles, opts.params, opts);
}

PressureResult RegisterPressureAnalysis::analyzeHypotheticalStatic(
    mlir::Region &region, llvm::ArrayRef<mlir::Operation *> clonedOps,
    const ArchHandler &arch, const ArchParams &archParams,
    const PressureQuery &opts) {
  mlir::Operation *anchor = region.getParentOp();
  mlir::Liveness localLiveness(anchor);

  PressureTrace trace;
  buildTrace(region, localLiveness, arch, archParams, trace);

  llvm::SmallVector<mlir::Value, 32> extraValues;
  llvm::DenseSet<mlir::Operation *> visited;
  collectOperandClosure(clonedOps, extraValues, visited);

  RegClassCounts extraCounts{};
  for (mlir::Value v : extraValues) {
    RegClassRequirement req = classifyAndCache(v, arch, archParams, trace);
    addValueToCounts(extraCounts, req);
  }
  for (PressurePoint &pt : trace.points) {
    for (unsigned i = 0; i < kNumRegClasses; ++i)
      pt.perClassLive[i] += extraCounts[i];
    updatePeak(trace.peakLive, pt.perClassLive);
    for (mlir::Value v : extraValues)
      pt.liveSet.push_back(v);
  }

  uint64_t spillCycles = aggregateSpills(opts.strategy, trace, opts.params,
                                          arch, archParams, opts.tripCount);
  return materialiseResult(trace, spillCycles, opts.params, opts);
}

} // namespace drcompiler
