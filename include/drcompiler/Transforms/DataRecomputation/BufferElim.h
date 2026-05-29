//===- BufferElim.h - Whole-buffer elimination verdicts --------*- C++ -*-===//
//
// Per-allocation-root verdicts for whether the buffer can be eliminated:
// escape state, count of remaining loads, classification rollup, and (when
// the rollup cost model has run) keep-vs-eliminate cost numbers.
//
//===----------------------------------------------------------------------===//

#ifndef DRCOMPILER_TRANSFORMS_DATARECOMPUTATION_BUFFERELIM_H
#define DRCOMPILER_TRANSFORMS_DATARECOMPUTATION_BUFFERELIM_H

#include "drcompiler/Analysis/ArchHandler.h"
#include "drcompiler/Analysis/RegisterClass.h"
#include "drcompiler/Analysis/SpillStrategy.h"
#include "drcompiler/Transforms/CpuCostModel.h"
#include "drcompiler/Transforms/DataRecomputation/AnalysisState.h"
#include "drcompiler/Transforms/DataRecomputation/CacheCostModel.h"
#include "drcompiler/Transforms/Utils/AllocEscapeAnalysis.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace dr {

/// Per-allocation-root counts snapshotted BEFORE per-load strategies run.
/// Strategies may erase load ops, so this snapshot is the only reliable
/// source for the original classification counts.
struct PreElimRootStats {
  unsigned loads = 0;
  unsigned stores = 0;
  unsigned multi = 0;
  unsigned leaked = 0;
  unsigned killed = 0;
};

struct BufferElimVerdict {
  mlir::Operation *allocRoot = nullptr;
  drcompiler::EscapeResult escape;
  unsigned loadCount = 0;       // total loads observed reaching this root
  unsigned storeCount = 0;      // total stores reaching this root
  unsigned remainingLoads = 0;  // loads still in IR post-strategies
  unsigned multiLoadCount = 0;
  unsigned leakedLoadCount = 0;
  unsigned killedLoadCount = 0;
  bool allLoadsReplaced = false;
  bool feasible = false;        // pre-cost feasibility verdict

  // Filled in by the rollup cost model (task 3). Left at defaults here.
  int64_t bufferSizeBytes = 0;
  unsigned keepCost = 0;
  unsigned elimCost = 0;
  bool costApproved = false;
};

/// Per-root SSA tree info supplied by the caller: for each alloc root, the
/// stored-value SSA root from each reaching store (so we can recompute ALU
/// cost and, later, share-discount/reg-pressure). Multiple entries when
/// distinct stores reach the same buffer.
using BufferStoredValues =
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Value, 4>>;

/// Per-buffer-elim tuning knobs (forwarded from pass options).
///
/// Register-pressure penalty is computed via RegisterPressureAnalysis when
/// `arch` is non-null.  Without an ArchHandler, the rollup falls back to a
/// zero register-pressure penalty (memory + ALU costs only).
struct BufferElimTuning {
  unsigned icacheSoftBudget = 128;
  const drcompiler::ArchHandler *arch = nullptr;
  drcompiler::ArchParams archParams;
  drcompiler::RegisterParams regParams;
  drcompiler::SpillStrategy spillStrategy = drcompiler::SpillStrategy::ExcessHot;
};

/// Single-buffer cost rollup. Computes keep/elim cycle estimates and a
/// boolean eliminate decision from the inputs the caller has already
/// collected. Pure with respect to the IR — does not walk uses.
BufferElimCostDecision
computeBufferElimCost(mlir::Operation *allocRoot,
                      const PreElimRootStats &stats,
                      llvm::ArrayRef<mlir::Value> storedValues,
                      const CacheParams &cache,
                      const drcompiler::CpuCostModel &cpu,
                      const BufferElimTuning &tuning);

/// Walk every alloc root in `allocRootFor`, decide feasibility based on
/// escape state + remaining loads + classification counts. `keepBuffers`
/// are roots the per-load cost model already vetoed — they cannot be
/// eliminated and are reported as infeasible. `liveLoads` is the set of
/// load ops that survived the per-load strategies.
///
/// When `cache` and `cpu` are supplied, also runs the rollup cost model
/// and fills in keep/elim/costApproved on each verdict. `storedValues`
/// supplies the SSA root of each reaching store per buffer.
llvm::SmallVector<BufferElimVerdict>
computeBufferElimVerdicts(
    mlir::ModuleOp moduleOp,
    AllocationRoots &allocRootFor,
    const llvm::DenseSet<mlir::Operation *> &keepBuffers,
    const llvm::DenseMap<mlir::Operation *, PreElimRootStats> &preStats,
    const BufferStoredValues &storedValues,
    const llvm::SmallDenseSet<mlir::Operation *> &liveLoads,
    mlir::SymbolTableCollection &symTab,
    const CacheParams &cache,
    const drcompiler::CpuCostModel &cpu,
    const BufferElimTuning &tuning);

} // namespace dr

#endif // DRCOMPILER_TRANSFORMS_DATARECOMPUTATION_BUFFERELIM_H
