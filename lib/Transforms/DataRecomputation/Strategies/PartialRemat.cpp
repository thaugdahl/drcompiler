//===- PartialRemat.cpp - Strategy 2b ------------------------------------===//
#include "drcompiler/Transforms/DataRecomputation/Strategies/PartialRemat.h"

#include "drcompiler/Transforms/Utils/OpDispatchUtils.h"

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace dr::strategies {

namespace {

const char *rejectReason(PartialLeafReject r) {
  switch (r) {
  case PartialLeafReject::None: return "none";
  case PartialLeafReject::NoAllocRoot: return "no-alloc-root";
  case PartialLeafReject::AllocaOutOfScope: return "alloca-out-of-scope";
  case PartialLeafReject::Escapes: return "escapes-to-call";
  case PartialLeafReject::MultiWrite: return "intervening-write";
  case PartialLeafReject::WriterDoesNotDominate:
    return "writer-does-not-dominate";
  case PartialLeafReject::OperandNotLive: return "index-not-live";
  }
  return "unknown";
}

} // namespace

Outcome PartialRemat::tryApply(LoadCandidate &c, StrategyEnv &env) {
  if (!env.partialRematEnabled)
    return Outcome::NotApplicable;

  llvm::SmallVector<mlir::Operation *> partialOps;
  llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> partialSubs;
  llvm::SmallVector<mlir::Operation *> partialLeaves;

  PartialRematOpts partial;
  partial.allow = true;
  partial.maxLeaves = env.partialMaxLeaves;
  partial.allocRootFor = &env.allocRootFor;
  partial.rootWrites = &env.rootWrites;
  partial.leaves = &partialLeaves;

  // Phase C: detect isomorphic enclosing for-nests between the producer
  // store and the consumer load. When matched and the index expressions
  // agree under the IV substitution, build an IRMapping that maps each
  // producer IV to its consumer counterpart. The remat clone path then
  // resolves producer-IV references to consumer IVs at clone time,
  // unblocking the sibling-loop recompute pattern in onnx-mlir output.
  mlir::IRMapping ivMapping;
  llvm::SmallVector<std::pair<mlir::Value, mlir::Value>, 4> ivPairs;
  bool useIVSub = false;
  if (mode == Mode::Intraprocedural && !argMapping &&
      matchEnclosingForNests(c.storeOp, c.loadOp, ivPairs) &&
      affineAccessEqualUnderIVSub(c.loadOp, c.storeOp, ivPairs)) {
    for (auto &kv : ivPairs)
      ivMapping.map(kv.first, kv.second);
    useIVSub = true;
    partial.ivSubstitution = true;
  }
  const mlir::IRMapping *effArgMapping = useIVSub ? &ivMapping : argMapping;

  if (!isRematerializable(c.storedValue, c.loadOp, env.dom, partialOps,
                          effArgMapping, &env.loadProv, &partialSubs,
                          &partial)) {
    if (env.emitDiagnostics &&
        partial.lastReject != PartialLeafReject::None) {
      c.loadOp->emitRemark()
          << "partial-remat: REJECT_UNSAFE (reason="
          << rejectReason(partial.lastReject) << ")";
    }
    if (env.summaryEnabled &&
        partial.lastReject != PartialLeafReject::None) {
      summaryLog(env, c.loadOp,
                 (llvm::Twine("partial-remat REJECT_UNSAFE (reason=") +
                  rejectReason(partial.lastReject) + ")").str());
    }
    return Outcome::NotApplicable;
  }

  // Cost gating differs between modes.
  unsigned alu = estimateComputeCost(c.storedValue, env.costModel);
  unsigned leaf = estimateLeafLoadsCost(partialLeaves, c.loadOp, env.cache,
                                        env.allocRootFor);
  mlir::Value loadMemref = drcompiler::getLoadStoreMemref(c.loadOp);
  mlir::Operation *loadRoot = lookupAllocRoot(loadMemref, env.allocRootFor);
  int64_t sizeBytes = (int64_t)env.cache.l2Size + 1;
  if (loadRoot)
    if (auto bs = estimateBufferSizeBytes(loadRoot))
      sizeBytes = *bs;
  unsigned loadLat =
      estimateAccessLatency(c.loadOp, c.loadOp, sizeBytes, env.cache);

  bool reject = false;
  if (mode == Mode::Intraprocedural) {
    // Strict gate: only fire if re-execution is cheaper than the original load.
    reject = (alu + leaf >= loadLat);
  } else {
    auto dec = decideBufferStrategy(alu, leaf, loadLat,
                                    /*numConsumers=*/1, sizeBytes,
                                    /*storeToLoadFootprint=*/0,
                                    /*operandPenalty=*/0, env.cache);
    reject = !dec.recompute;
  }
  if (reject) {
    if (env.emitDiagnostics)
      c.loadOp->emitRemark()
          << "partial-remat: REJECT_COST (alu=" << alu
          << ", leaves=" << leaf << ", keep=" << loadLat << ")";
    summaryLog(env, c.loadOp,
               (llvm::Twine("partial-remat REJECT_COST (alu=") +
                llvm::Twine(alu) + ", leaves=" + llvm::Twine(leaf) +
                ", keep=" + llvm::Twine(loadLat) + ")").str());
    return Outcome::Accepted; // matches original `continue` semantics
  }

  mlir::Value clonedVal = rematerializeAt(c.storedValue, c.loadOp, partialOps,
                                          effArgMapping, partialSubs);
  if (clonedVal.getType() != c.loadOp->getResult(0).getType()) {
    if (env.emitDiagnostics)
      c.loadOp->emitRemark() << "partial-remat: REJECT_TYPE";
    summaryLog(env, c.loadOp, "partial-remat REJECT_TYPE");
    return Outcome::Accepted;
  }
  if (env.emitDiagnostics)
    c.loadOp->emitRemark()
        << "partial-remat: ACCEPT (alu=" << alu << ", leaves=" << leaf
        << ", load=" << loadLat << ")";
  summaryLog(env, c.loadOp,
             (llvm::Twine("partial-remat ACCEPT (alu=") + llvm::Twine(alu) +
              ", leaves=" + llvm::Twine(leaf) + ", load=" +
              llvm::Twine(loadLat) + ")").str());
  c.loadOp->getResult(0).replaceAllUsesWith(clonedVal);
  c.loadOp->erase();
  return Outcome::Accepted;
}

} // namespace dr::strategies
