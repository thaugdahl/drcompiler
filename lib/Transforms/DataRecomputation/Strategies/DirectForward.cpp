//===- DirectForward.cpp - Strategy 1 ------------------------------------===//
#include "drcompiler/Transforms/DataRecomputation/Strategies/DirectForward.h"

#include "mlir/IR/Operation.h"

namespace dr::strategies {

Outcome DirectForward::tryApply(LoadCandidate &c, StrategyEnv &env) {
  // Same-function check — caller already filters to same-fn pairs, but
  // guard defensively.
  if (c.loadOp->getParentOp() == nullptr || c.storeOp->getParentOp() == nullptr)
    return Outcome::NotApplicable;

  if (!env.dom.dominates(c.storeOp, c.loadOp))
    return Outcome::NotApplicable;

  if (c.storedValue.getType() != c.loadOp->getResult(0).getType()) {
    if (env.emitDiagnostics)
      c.loadOp->emitRemark() << "direct-forward: REJECT_TYPE";
    summaryLog(env, c.loadOp, "direct-forward REJECT_TYPE");
    // Type-mismatch is treated as "consumed" — original code uses `continue`
    // to skip subsequent strategies in the same iteration.
    return Outcome::Accepted;
  }

  if (env.emitDiagnostics)
    c.loadOp->emitRemark() << "direct-forward: ACCEPT";
  summaryLog(env, c.loadOp, "direct-forward ACCEPT");
  c.loadOp->getResult(0).replaceAllUsesWith(c.storedValue);
  c.loadOp->erase();
  return Outcome::Accepted;
}

} // namespace dr::strategies
