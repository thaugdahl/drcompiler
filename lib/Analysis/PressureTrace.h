//===- PressureTrace.h - Internal trace shared by spill strategies ---------===//
//
// This header is private to lib/Analysis/.  It carries the per-program-point
// liveness data that all three SpillStrategy implementations consume.
//
//===----------------------------------------------------------------------===//

#ifndef DRCOMPILER_LIB_ANALYSIS_PRESSURETRACE_H
#define DRCOMPILER_LIB_ANALYSIS_PRESSURETRACE_H

#include "drcompiler/Analysis/ArchHandler.h"
#include "drcompiler/Analysis/RegisterClass.h"
#include "drcompiler/Analysis/RegisterPressureAnalysis.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace drcompiler {

/// One program point in the trace.  We materialise the set of values that
/// are live across this op so the GraphColor strategy can derive interference
/// edges; ExcessHot/SumExcess only need the per-class counts.
struct PressurePoint {
  mlir::Operation *op = nullptr;
  RegClassCounts perClassLive{};
  llvm::SmallVector<mlir::Value, 16> liveSet;
};

/// Sequence of points + per-value classification, in program order over the
/// region under analysis.
struct PressureTrace {
  llvm::SmallVector<PressurePoint, 32> points;

  /// Region-wide per-class peak; cached after construction so strategies
  /// don't have to recompute it.
  RegClassCounts peakLive{};

  /// Cached classification per value.  ExcessHot only needs peak; SumExcess
  /// walks `points`; GraphColor needs both per-value class and live sets.
  llvm::DenseMap<mlir::Value, RegClassRequirement> classification;
};

/// Strategy entry points.  Each returns the total spill-cycle estimate
/// for the trace under the given parameters.
uint64_t aggregateExcessHot(const PressureTrace &trace,
                            const RegisterParams &params,
                            const ArchHandler &arch,
                            const ArchParams &archParams,
                            uint64_t tripCount);

uint64_t aggregateSumExcess(const PressureTrace &trace,
                            const RegisterParams &params,
                            const ArchHandler &arch,
                            const ArchParams &archParams);

uint64_t aggregateGraphColor(const PressureTrace &trace,
                             const RegisterParams &params,
                             const ArchHandler &arch,
                             const ArchParams &archParams);

} // namespace drcompiler

#endif // DRCOMPILER_LIB_ANALYSIS_PRESSURETRACE_H
