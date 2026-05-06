//===- InterprocRemat.cpp - Strategy 3 -----------------------------------===//
#include "drcompiler/Transforms/DataRecomputation/Strategies/InterprocRemat.h"
#include "drcompiler/Transforms/DataRecomputation/Strategies/PartialRemat.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace dr::strategies {

Outcome InterprocRemat::tryApply(LoadCandidate &c, StrategyEnv &env) {
  auto origIt = env.interprocOrigins.find(c.storeOp);
  if (origIt == env.interprocOrigins.end())
    return Outcome::NotApplicable;

  if (origIt->second.size() > 1) {
    if (env.emitDiagnostics)
      c.loadOp->emitRemark() << "interproc: SKIP_AMBIGUOUS_ORIGIN";
    summaryLog(env, c.loadOp, "interproc SKIP_AMBIGUOUS_ORIGIN");
    return Outcome::Accepted; // matches original `continue` semantics
  }

  for (auto &origin : origIt->second) {
    if (!env.dom.dominates(origin.callSiteOp, c.loadOp))
      continue;

    mlir::Region *calleeBody = origin.callee.getCallableRegion();
    if (!calleeBody || calleeBody->empty())
      continue;

    mlir::IRMapping argMapping;
    for (auto [calleeArg, callOperand] :
         llvm::zip(calleeBody->getArguments(),
                   origin.callSiteOp->getOperands()))
      argMapping.map(calleeArg, callOperand);

    llvm::SmallVector<mlir::Operation *> opsToClone;
    llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> loadSubs;
    if (isRematerializable(c.storedValue, c.loadOp, env.dom, opsToClone,
                           &argMapping, &env.loadProv, &loadSubs)) {
      mlir::Value clonedVal = rematerializeAt(c.storedValue, c.loadOp,
                                              opsToClone, &argMapping,
                                              loadSubs);
      if (clonedVal.getType() != c.loadOp->getResult(0).getType()) {
        if (env.emitDiagnostics)
          c.loadOp->emitRemark() << "interproc-remat: REJECT_TYPE";
        summaryLog(env, c.loadOp, "interproc-remat REJECT_TYPE");
        return Outcome::Accepted;
      }
      if (env.emitDiagnostics)
        c.loadOp->emitRemark() << "interproc-remat: ACCEPT";
      summaryLog(env, c.loadOp, "interproc-remat ACCEPT");
      c.loadOp->getResult(0).replaceAllUsesWith(clonedVal);
      c.loadOp->erase();
      return Outcome::Accepted;
    }
    if (env.emitDiagnostics)
      c.loadOp->emitRemark() << "interproc-remat: REJECT_UNSAFE";
    summaryLog(env, c.loadOp, "interproc-remat REJECT_UNSAFE");

    // Per-origin partial-remat fallback (interprocedural mode).
    PartialRemat partial(PartialRemat::Mode::Interprocedural, &argMapping);
    if (partial.tryApply(c, env) == Outcome::Accepted)
      return Outcome::Accepted;
  }
  return Outcome::NotApplicable;
}

} // namespace dr::strategies
