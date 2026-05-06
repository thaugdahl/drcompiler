//===- CrossFnOrdered.h - Strategy 4 family --------------------------------===//
//
//===----------------------------------------------------------------------===//
//
// Strategy 4: cross-function ordered recomputation. Activates when load and
// store are in different functions and a common caller orders the writer
// call before the reader call with no intervening clobber. Materializes the
// writer's expression at the caller and threads the result into the reader
// as an extra arg (in-place when the reader is private + single-caller,
// otherwise via a private "__dr_spec_N" clone).
//
// Sub-strategies dispatched by the plan builder:
//   * Straight-line cross-fn (the original Strategy 4)
//   * Strategy D — chained sub-plans for inner loads
//   * Strategy F.1 — single-iteration extraction
//   * Strategy F.2 — full loop clone into a scratch buffer
//
//===----------------------------------------------------------------------===//

#ifndef DRCOMPILER_TRANSFORMS_DATARECOMPUTATION_STRATEGIES_CROSSFNORDERED_H
#define DRCOMPILER_TRANSFORMS_DATARECOMPUTATION_STRATEGIES_CROSSFNORDERED_H

#include "drcompiler/Transforms/DataRecomputation/AnalysisState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>
#include <optional>

namespace dr::strategies {

struct CrossFnRematPlan;

/// Recursive sub-plan record (Strategy D — chained loads).
struct LoadSubRecord {
  mlir::Operation *loadOp = nullptr;
  mlir::Operation *innerStoreOp = nullptr;
  mlir::Operation *innerGlobalOp = nullptr;
  mlir::FunctionOpInterface innerWriterFn{};
  std::unique_ptr<CrossFnRematPlan> subPlan;

  LoadSubRecord();
  LoadSubRecord(LoadSubRecord &&) noexcept;
  LoadSubRecord &operator=(LoadSubRecord &&) noexcept;
  ~LoadSubRecord();
};

/// Loop-cloning materialization record (Strategy F).
struct LoopMatRecord {
  mlir::Operation *outerLoopOp = nullptr;
  mlir::Operation *storeOp = nullptr;
  mlir::Operation *globalOp = nullptr;
  mlir::Operation *getGlobalOp = nullptr;
  bool boundsAreStatic = false;
  bool extractionViable = false;
  llvm::SmallVector<mlir::BlockArgument, 2> storeIVs;
};

/// Cross-function rematerialization plan.
struct CrossFnRematPlan {
  llvm::SmallVector<mlir::Operation *, 4> opsToClone;
  llvm::SmallVector<mlir::BlockArgument, 4> writerArgs;
  llvm::SmallVector<mlir::Operation *, 4> hoistedLoadGlobals;
  llvm::SmallVector<LoadSubRecord, 2> loadSubs;
  std::optional<LoopMatRecord> loopMat;
  mlir::Value rootValue;
};

/// Recursion cap for chained sub-plans (Strategy D).
constexpr unsigned kMaxCrossFnChainDepth = 3;

/// Caller-side analysis state needed for plan-build, caller selection,
/// and materialization.
struct MatCtx {
  mlir::DominanceInfo *domInfo = nullptr;
  const ModuleGlobalWrites *moduleGlobalWrites = nullptr;
  AllocationRoots *allocRootFor = nullptr;
  const llvm::DenseSet<mlir::Operation *> *globalAllocs = nullptr;
  llvm::function_ref<bool(mlir::Operation *, mlir::Operation *)>
      fnWritesGlobal;
};

/// Plan-builder. Returns true when the writer's stored value tree can be
/// materialized at a caller.
bool buildCrossFunctionRematPlan(
    mlir::Operation *storeOp,
    mlir::FunctionOpInterface writerFn,
    AllocationRoots &allocRootFor,
    LoadProvenanceMap &loadProv,
    llvm::function_ref<bool(mlir::Operation *, mlir::Operation *)>
        fnWritesGlobal,
    CrossFnRematPlan &plan,
    unsigned depth,
    llvm::SmallDenseSet<mlir::Operation *> seenStores);

/// Find the latest call to writerFn in caller that strictly dominates
/// readerCall and has no intervening writer to globalOp between them.
mlir::Operation *findEligibleWriterCall(
    mlir::FunctionOpInterface caller,
    mlir::Operation *readerCall,
    mlir::FunctionOpInterface writerFn,
    mlir::Operation *globalOp,
    mlir::DominanceInfo &domInfo,
    const ModuleGlobalWrites &moduleGlobalWrites,
    AllocationRoots &allocRootFor,
    const llvm::DenseSet<mlir::Operation *> &globalAllocOps,
    llvm::function_ref<bool(mlir::Operation *, mlir::Operation *)>
        fnWritesGlobal);

/// Materialize the writer's expression at the caller, just after writerCall.
mlir::Value materializeAtCaller(mlir::Operation *writerCall,
                                const CrossFnRematPlan &plan,
                                mlir::OpBuilder &builder,
                                const MatCtx &ctx);

/// Strategy F.2 — allocate a scratch buffer at the caller and clone the
/// writer's outer loop into it.
mlir::Value materializeLoop(mlir::Operation *writerCall,
                            const CrossFnRematPlan &plan,
                            mlir::OpBuilder &builder,
                            const MatCtx &ctx);

/// Strategy F.1 — clone one iteration's worth of the writer's loop body
/// at the caller, with each loop IV substituted by idxValues. Returns the
/// cloned stored value as a scalar.
mlir::Value materializeExtraction(mlir::Operation *writerCall,
                                  const CrossFnRematPlan &plan,
                                  mlir::OpBuilder &builder,
                                  llvm::ArrayRef<mlir::Value> idxValues);

/// Reader-side rewrites.
mlir::BlockArgument addParamAndReplaceLoads(
    mlir::FunctionOpInterface funcOp,
    mlir::Type extraType,
    llvm::ArrayRef<mlir::Operation *> loadOps);

mlir::BlockArgument addMemrefParamAndRewriteLoads(
    mlir::FunctionOpInterface funcOp,
    mlir::Type memrefTy,
    llvm::ArrayRef<mlir::Operation *> loadOps);

mlir::FunctionOpInterface cloneReaderForSpec(
    mlir::FunctionOpInterface reader);

void rewriteCallSite(mlir::Operation *oldCall,
                     mlir::FunctionOpInterface newCallee,
                     llvm::ArrayRef<mlir::Value> extraOperands);

} // namespace dr::strategies

#endif // DRCOMPILER_TRANSFORMS_DATARECOMPUTATION_STRATEGIES_CROSSFNORDERED_H
