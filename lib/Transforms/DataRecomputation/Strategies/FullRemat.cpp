//===- FullRemat.cpp - Strategy 2 ----------------------------------------===//
#include "drcompiler/Transforms/DataRecomputation/Strategies/FullRemat.h"

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace dr::strategies {

Outcome FullRemat::tryApply(LoadCandidate &c, StrategyEnv &env) {
  llvm::SmallVector<mlir::Operation *> opsToClone;
  llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> loadSubs;

  if (!isRematerializable(c.storedValue, c.loadOp, env.dom, opsToClone,
                          /*argMapping=*/nullptr, &env.loadProv,
                          &loadSubs)) {
    if (env.emitDiagnostics)
      c.loadOp->emitRemark() << "full-remat: REJECT_UNSAFE";
    summaryLog(env, c.loadOp, "full-remat REJECT_UNSAFE");
    return Outcome::NotApplicable;
  }

  mlir::Value clonedVal = rematerializeAt(c.storedValue, c.loadOp, opsToClone,
                                          /*argMapping=*/nullptr, loadSubs);
  if (clonedVal.getType() != c.loadOp->getResult(0).getType()) {
    if (env.emitDiagnostics)
      c.loadOp->emitRemark() << "full-remat: REJECT_TYPE";
    summaryLog(env, c.loadOp, "full-remat REJECT_TYPE");
    return Outcome::Accepted; // matches original `continue` semantics
  }

  if (env.emitDiagnostics)
    c.loadOp->emitRemark() << "full-remat: ACCEPT";
  summaryLog(env, c.loadOp, "full-remat ACCEPT");
  c.loadOp->getResult(0).replaceAllUsesWith(clonedVal);
  c.loadOp->erase();
  return Outcome::Accepted;
}

} // namespace dr::strategies
