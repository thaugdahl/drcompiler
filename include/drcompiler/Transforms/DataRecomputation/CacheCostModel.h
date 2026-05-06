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

#include "drcompiler/Transforms/CpuCostModel.h"
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
struct CacheParams {
  unsigned l1Size;        // bytes
  unsigned l2Size;        // bytes
  unsigned l3Size;        // bytes (0 = unknown / not modeled)
  unsigned l1Latency;     // cycles
  unsigned l2Latency;
  unsigned l3Latency;
  unsigned memLatency;
  unsigned cacheLineSize; // bytes
};

/// Estimate the ALU cost of recomputing a value by walking its SSA operand
/// tree. Each operation is weighted via the CpuCostModel. Loads and block
/// arguments are free (they are inputs, not recomputed).
unsigned estimateComputeCost(mlir::Value val,
                             const drcompiler::CpuCostModel &costModel);

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

/// Per-buffer materialization decision.
struct MaterializationDecision {
  bool recompute;               // true = eliminate buffer, false = keep it
  unsigned aluCost;             // ALU cost to recompute one element
  unsigned leafLoadCost;        // per-element cost of cloned partial leaf loads
  unsigned loadLatency;         // estimated load latency (effective)
  unsigned numConsumers;        // number of SINGLE loads from this buffer
  int64_t bufferSizeBytes;
  int64_t storeToLoadFootprint; // intervening bytes between store and load
  unsigned operandPenalty;      // per-element operand reload penalty (cycles)
};

/// Decide whether to recompute or keep a buffer. With leafLoadCost = 0
/// and footprint analysis disabled this reduces to:
///   keepCost      = aluCost + 1 + numConsumers * loadLatency
///   recomputeCost = numConsumers * aluCost
MaterializationDecision decideBufferStrategy(unsigned aluCost,
                                             unsigned leafLoadCost,
                                             unsigned loadLatency,
                                             unsigned numConsumers,
                                             int64_t bufferSizeBytes,
                                             int64_t storeToLoadFootprint,
                                             unsigned operandPenalty,
                                             const CacheParams &cache);

} // namespace dr

#endif // DRCOMPILER_TRANSFORMS_DATARECOMPUTATION_CACHECOSTMODEL_H
