//===- DirectForward.h - Strategy 1: same-fn store-dominates-load --------===//
#ifndef DRCOMPILER_TRANSFORMS_DATARECOMPUTATION_STRATEGIES_DIRECTFORWARD_H
#define DRCOMPILER_TRANSFORMS_DATARECOMPUTATION_STRATEGIES_DIRECTFORWARD_H

#include "drcompiler/Transforms/DataRecomputation/Strategies/Strategy.h"

namespace dr::strategies {

/// Same-function direct value forwarding: when the store dominates the
/// load, replace the load's result with the store's stored value.
class DirectForward final : public LoadStrategy {
public:
  llvm::StringRef name() const override { return "direct-forward"; }
  Outcome tryApply(LoadCandidate &c, StrategyEnv &env) override;
};

} // namespace dr::strategies

#endif
