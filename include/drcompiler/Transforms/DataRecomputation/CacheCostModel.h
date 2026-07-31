//===- CacheCostModel.h - Cache-aware cost model for DataRecomputation ----===//
//
//===----------------------------------------------------------------------===//
//
// Cache hierarchy parameters and footprint-based cost helpers used by the
// DataRecomputation pass to decide whether to keep a buffer (load) or
// recompute its stored value.
//
// All entry points are pure with respect to the IR (no mutation).
//
//===----------------------------------------------------------------------===//

#ifndef DRCOMPILER_TRANSFORMS_DATARECOMPUTATION_CACHECOSTMODEL_H
#define DRCOMPILER_TRANSFORMS_DATARECOMPUTATION_CACHECOSTMODEL_H

#include "drcompiler/Analysis/ArchHandler.h"
#include "drcompiler/Analysis/CpuCostModel.h"
#include "drcompiler/Transforms/DataRecomputation/AnalysisState.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseSet.h"

#include <cstdint>
#include <optional>

namespace dr {

/// Default trip count when bounds cannot be statically determined.
/// Picked to match a common per-loop iteration count seen in SPEC inner
/// loops: big enough that small constants (1, 4) do not dominate cache
/// estimates, small enough that footprint x trip stays within typical
/// L2 sizes.
constexpr int64_t kDefaultTripCount = 128;

/// Cache hierarchy parameters for the cost model.
///
/// `llcSharers` makes the model contention-aware: the shared last-level cache is
/// not exclusively ours, so its *effective* capacity is derated by the number of
/// cores/processes that can evict our lines (private L1/L2 are not derated). A
/// reuse whose distance fits the physical L3 but not `l3Size / llcSharers` is
/// priced as a memory access — the residency a co-tenant can take away. Default
/// 1 reproduces the exclusive-cache (isolated) model.
struct CacheParams {
  unsigned l1Size;        // bytes (private)
  unsigned l2Size;        // bytes (private on current x86)
  unsigned l3Size;        // bytes (0 = unknown / not modeled; shared LLC)
  unsigned l1Latency;     // cycles
  unsigned l2Latency;
  unsigned l3Latency;
  unsigned memLatency;
  unsigned cacheLineSize; // bytes
  unsigned llcSharers = 1; // co-tenants of the shared LLC (effective L3 = l3Size/this)
  unsigned l2OccupancyPct = 100; // fraction of the private L2 a working set may use
                                 // and still be treated as L2-resident (margin for
                                 // co-resident arrays / SMT / prefetch). 100 = no
                                 // derate (DR default); fission sets it lower.
};

/// Estimate the ALU cost of recomputing a value by walking its SSA operand
/// tree. Each operation is weighted via the CpuCostModel. Loads and block
/// arguments are free (they are inputs, not recomputed).
///
/// `issueWidth` is the target's sustained superscalar issue width
/// (ArchParams::issueWidth): the result is the max of the dependency critical
/// path and the op count spread over that many issue slots, so a wide
/// INDEPENDENT cone is cheaper on a wide machine and a dependent chain is
/// insensitive to it. Callers that have already resolved an ArchHandler pass
/// `archParams.issueWidth`; the default matches the generic handler (and the
/// `kIssueWidth = 4` literal this parameter replaced).
unsigned estimateComputeCost(mlir::Value val,
                             const drcompiler::CpuCostModel &costModel,
                             unsigned issueWidth = 4);

/// Estimate the size of an allocation in bytes. Returns nullopt when the
/// size cannot be determined statically (non-alloc op, dynamic shape,
/// zero-bit element type).
std::optional<int64_t> estimateBufferSizeBytes(mlir::Operation *allocOp);

/// Estimate the cost of one load from a buffer of given size, given the
/// cache hierarchy.
unsigned estimateLoadLatency(int64_t bufferSizeBytes,
                             const CacheParams &cache);

/// Trace a value to a compile-time constant integer, walking through
/// index_cast ops and (one level of) call-site argument forwarding.
std::optional<int64_t> traceToConstant(mlir::Value val,
                                       const EnrichedCallGraph &callGraph);

/// Estimate the trip count of an affine.for or scf.for. Returns nullopt
/// if the trip count cannot be determined.
std::optional<int64_t> estimateTripCount(mlir::Operation *loopOp,
                                         const EnrichedCallGraph &callGraph);

/// Find the innermost enclosing affine.for / scf.for induction variable of
/// `op`, or a null Value when `op` is not inside any such loop.
mlir::Value innermostEnclosingIV(mlir::Operation *op);

/// Element size in bytes of a memref load/store op (8 if non-int/float).
unsigned accessElementBytes(mlir::Operation *accessOp);

/// Estimate the per-iteration address stride (in ELEMENTS) of a load/store
/// relative to induction variable `iv`:
///   - 0 when the access is invariant in iv (same address every iteration),
///   - a positive integer when statically determinable,
///   - std::nullopt when it cannot be determined (callers should pessimize as
///     full-cache-line, i.e. spatially non-local).
/// Handles both memref.load/store and affine.load/store.
std::optional<int64_t> estimateAccessStrideElements(mlir::Operation *accessOp,
                                                     mlir::Value iv);

/// Estimate the total memory footprint of operations between storeOp and
/// loadOp in program order. This is the "interjected" memory traffic
/// that determines whether the stored value is still in cache when the
/// load runs.
int64_t estimateInterveningFootprint(mlir::Operation *storeOp,
                                     mlir::Operation *loadOp,
                                     const CacheParams &cache,
                                     const EnrichedCallGraph &callGraph);

/// Collect the memref values that the SSA operand tree of `val` loads
/// from. These are the operands that would need to be re-loaded during
/// recomputation.
void collectOperandMemrefs(mlir::Value val,
                           llvm::SmallDenseSet<mlir::Value> &memrefs);

/// Estimate the operand reload penalty for recomputation. Walks the SSA
/// operand tree of the stored value to find which memrefs recomputation
/// would need to read; for each, checks whether that memref is
/// re-accessed in the intervening ops between store and load. Returns a
/// per-element penalty in cycles to be added to the recomputation cost.
unsigned estimateOperandReloadPenalty(mlir::Value storedVal,
                                      mlir::Operation *storeOp,
                                      mlir::Operation *loadOp,
                                      int64_t storeToLoadFootprint,
                                      const CacheParams &cache);

/// Inputs to per-buffer materialization decision.  The register-pressure
/// component (`regCyclesKeep`/`regCyclesRecompute`) is supplied by the
/// caller from a RegisterPressureAnalysis query; this struct intentionally
/// does not reach into MLIR IR.
struct MaterializationInputs {
  unsigned aluCost = 0;
  unsigned leafLoadCost = 0;
  unsigned loadLatency = 0;
  unsigned numConsumers = 0;
  int64_t bufferSizeBytes = 0;
  int64_t storeToLoadFootprint = 0;
  unsigned operandPenalty = 0;
  unsigned regCyclesKeep = 0;
  unsigned regCyclesRecompute = 0;
};

/// Per-buffer materialization decision.  Carries the per-aspect breakdown
/// so diagnostics can show why the cost model went one way or the other.
struct MaterializationDecision {
  bool recompute = false;       // true = eliminate buffer, false = keep it
  // Echo of the inputs, kept for diagnostic back-compat.
  unsigned aluCost = 0;
  unsigned leafLoadCost = 0;
  unsigned loadLatency = 0;
  unsigned numConsumers = 0;
  int64_t bufferSizeBytes = 0;
  int64_t storeToLoadFootprint = 0;
  unsigned operandPenalty = 0;
  // Per-aspect cycle breakdown.
  unsigned memCyclesKeep = 0, memCyclesRecompute = 0;
  unsigned regCyclesKeep = 0, regCyclesRecompute = 0;
  unsigned aluCyclesKeep = 0, aluCyclesRecompute = 0;
  // Combined totals via ArchHandler::combineCosts.
  unsigned totalKeep = 0, totalRecompute = 0;
};

/// Decide whether to recompute or keep a buffer using the unified cost
/// model (memory + register pressure + ALU, combined per-arch).
///
///   memKeep      = numConsumers * effectiveLoadLatency
///   aluKeep      = aluCost + 1                              // store once
///   regKeep      = inputs.regCyclesKeep
///   memRecompute = numConsumers * (leafLoadCost + operandPenalty)
///   aluRecompute = numConsumers * aluCost
///   regRecompute = inputs.regCyclesRecompute
///
/// The keep / recompute totals are produced by `arch.combineCosts(...)`.
MaterializationDecision
decideBufferStrategy(const MaterializationInputs &inputs,
                     const CacheParams &cache,
                     const drcompiler::ArchHandler &arch,
                     const drcompiler::ArchParams &archParams);

/// Inputs to the whole-buffer elimination cost rollup. Treats the alloc
/// as the cost unit (vs the per-load `decideBufferStrategy`).
struct BufferElimCostInputs {
  int64_t bufferSizeBytes = 0;    // 0 = dynamic → use l2Size+1 pessimistically
  unsigned numLoads = 0;
  unsigned numStores = 0;
  unsigned loadLatency = 0;       // cycles per load (caller resolves vs cache)
  unsigned storeLatency = 0;      // cycles per store
  unsigned allocOverheadCycles = 0;   // amortized malloc/free cost; 0 for alloca
  unsigned capacityPenaltyCycles = 0; // pressure cost from this buffer
  unsigned perElemComputeCost = 0;    // ALU cycles to recompute one element
  unsigned numDistinctComputes = 0;   // distinct (parentFn, structHash) groups
                                      // across loads; ≤ numLoads. Optimal-CSE
                                      // assumption: per group, only one compute
                                      // pays; the rest are folded away.
  unsigned codeBloatPenalty = 0;      // icache pressure penalty
  unsigned regPressurePenalty = 0;    // spill penalty
};

struct BufferElimCostDecision {
  bool eliminate = false;
  unsigned keepCost = 0;
  unsigned elimCost = 0;
};

/// Cost rollup for whole-buffer elimination.
///   keepCost = numLoads*loadLatency + numStores*storeLatency
///              + allocOverhead + capacityPenalty
///   elimCost = max(0, totalRematCompute - sharedSubexprDiscount)
///              + codeBloatPenalty + regPressurePenalty
/// Decision is `elimCost <= keepCost`.
BufferElimCostDecision
decideBufferElimination(const BufferElimCostInputs &inputs);

} // namespace dr

#endif // DRCOMPILER_TRANSFORMS_DATARECOMPUTATION_CACHECOSTMODEL_H
