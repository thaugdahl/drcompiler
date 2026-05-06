//===- RematKernel.h - Rematerialization analysis & cloning kernel --------===//
//
//===----------------------------------------------------------------------===//
//
// Pure-IR helpers used by every per-load rematerialization strategy:
//
//   * isRematerializable / rematerializeAt — the operand-tree walker that
//     decides whether the SSA chain rooted at a stored value can be cloned
//     just before a load and, if so, performs the cloning.
//
//   * Stride-aware leaf cost (estimateAccessLatency / estimateLeafLoadsCost).
//     Used by the partial-remat cost gate and by the cross-function plan
//     builder.
//
//   * Module-wide write summary (RootWriteMap / buildRootWriteMap) and the
//     PartialRematOpts knob struct that drive partial rematerialization
//     leaf-safety checks.
//
//===----------------------------------------------------------------------===//

#ifndef DRCOMPILER_TRANSFORMS_DATARECOMPUTATION_REMATKERNEL_H
#define DRCOMPILER_TRANSFORMS_DATARECOMPUTATION_REMATKERNEL_H

#include "drcompiler/Transforms/DataRecomputation/AnalysisState.h"
#include "drcompiler/Transforms/DataRecomputation/CacheCostModel.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace dr {

/// Maximum number of load→store chains to follow during rematerialization.
constexpr unsigned kMaxChainDepth = 8;

/// Maximum number of ops that may be cloned for a single rematerialization.
/// Prevents blowup on deeply chained arithmetic (e.g. SHA-256 rounds).
constexpr unsigned kMaxRematOps = 64;

/// Look up an allocation root for a load-like op's memref value. Traces
/// through view-like casts so subview/cast users resolve to their root.
mlir::Operation *lookupAllocRoot(
    mlir::Value memref,
    llvm::DenseMap<mlir::Value, mlir::Operation *> &allocRootFor);

/// Stride-aware per-issuance cost for a load/store at its current (or
/// intended) site. Scales the cold-miss latency by the fraction of a
/// cache line touched per iteration of the innermost enclosing loop at
/// `siteOp`. When there is no enclosing loop, or the stride cannot be
/// determined, the full miss latency is charged.
unsigned estimateAccessLatency(mlir::Operation *accessOp,
                               mlir::Operation *siteOp,
                               int64_t bufferBytes,
                               const CacheParams &cache);

/// Estimate the per-issuance cost of partial-leaf loads cloned during
/// partial rematerialization.
unsigned estimateLeafLoadsCost(
    llvm::ArrayRef<mlir::Operation *> partialLeaves,
    mlir::Operation *insertionPoint,
    const CacheParams &cache,
    llvm::DenseMap<mlir::Value, mlir::Operation *> &allocRootFor);

/// Per-allocation-root write summary used by partial rematerialization.
struct RootWriteSummary {
  unsigned writeCount = 0;
  mlir::Operation *singleWriter = nullptr;
  bool escapesToCall = false;
};

using RootWriteMap = llvm::DenseMap<mlir::Operation *, RootWriteSummary>;

/// Build a write summary per allocation root, covering every function in
/// the module.
RootWriteMap buildRootWriteMap(
    mlir::ModuleOp moduleOp,
    llvm::DenseMap<mlir::Value, mlir::Operation *> &allocRootFor);

/// Why a partial leaf candidate was rejected. Consumed by diagnostics.
enum class PartialLeafReject {
  None,
  NoAllocRoot,
  AllocaOutOfScope,
  Escapes,
  MultiWrite,
  WriterDoesNotDominate,
  OperandNotLive,
};

/// Options controlling the partial-rematerialization path in
/// isRematerializable. When enabled, non-chainable leaf loads may be
/// cloned verbatim if they pass the leaf-safety check and the leaf
/// budget has room.
struct PartialRematOpts {
  bool allow = false;
  /// When true, the leaf-load branch recurses into the load's INDEX
  /// operands so the index-computing SSA chain is cloned at the
  /// consumer site (not just the load itself). The accompanying
  /// argMapping must populate any producer-region block args (loop
  /// IVs) with their substituted consumer values. The memref operand
  /// is verified via `isValueAvailableAt` rather than recursed.
  bool ivSubstitution = false;
  unsigned maxLeaves = 0;
  llvm::DenseMap<mlir::Value, mlir::Operation *> *allocRootFor = nullptr;
  const RootWriteMap *rootWrites = nullptr;
  llvm::SmallVectorImpl<mlir::Operation *> *leaves = nullptr;
  PartialLeafReject lastReject = PartialLeafReject::None;
};

/// Detect isomorphic enclosing affine.for nests between two ops. Pairs
/// the matched IVs (innermost first). Returns true and populates
/// `ivPairs` (producer → consumer) when both ops sit at the same depth
/// in nests with matching constant trip counts and steps.
bool matchEnclosingForNests(
    mlir::Operation *producerOp, mlir::Operation *consumerOp,
    llvm::SmallVectorImpl<std::pair<mlir::Value, mlir::Value>> &ivPairs);

/// True when the consumer affine load addresses the same memory cell
/// as the producer affine store, modulo substituting producer IVs with
/// consumer IVs per `ivPairs`. Performs a syntactic affine-map equality
/// check followed by per-operand equivalence (allowing constant fold).
bool affineAccessEqualUnderIVSub(
    mlir::Operation *consumerLoad, mlir::Operation *producerStore,
    llvm::ArrayRef<std::pair<mlir::Value, mlir::Value>> ivPairs);

/// Decide whether the SSA operand tree rooted at rootVal can be cloned
/// (rematerialized) just before insertionPoint. On success, opsToClone
/// is filled in topological (def-before-use) order.
///
/// argMapping: optional callee-entry-arg → caller-value mapping for
/// interprocedural use.
///
/// loadProv + loadSubs: when both are non-null, enables chaining through
/// SINGLE-provenance loads. Each chained substitution is recorded in
/// loadSubs as (loadResult, storeValue).
bool isRematerializable(
    mlir::Value rootVal, mlir::Operation *insertionPoint,
    mlir::DominanceInfo &domInfo,
    llvm::SmallVectorImpl<mlir::Operation *> &opsToClone,
    const mlir::IRMapping *argMapping = nullptr,
    const LoadProvenanceMap *loadProv = nullptr,
    llvm::SmallVectorImpl<std::pair<mlir::Value, mlir::Value>> *loadSubs =
        nullptr,
    PartialRematOpts *partial = nullptr);

/// Clone the ops in opsToClone at insertionPoint and return the
/// rematerialized value corresponding to originalVal.
mlir::Value rematerializeAt(
    mlir::Value originalVal, mlir::Operation *insertionPoint,
    llvm::ArrayRef<mlir::Operation *> opsToClone,
    const mlir::IRMapping *argMapping = nullptr,
    llvm::ArrayRef<std::pair<mlir::Value, mlir::Value>> loadSubs = {});

} // namespace dr

#endif // DRCOMPILER_TRANSFORMS_DATARECOMPUTATION_REMATKERNEL_H
