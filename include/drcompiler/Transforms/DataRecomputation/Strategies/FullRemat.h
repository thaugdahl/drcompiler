//===- FullRemat.h - Strategy 2: same-fn full rematerialization ----------===//
#ifndef DRCOMPILER_TRANSFORMS_DATARECOMPUTATION_STRATEGIES_FULLREMAT_H
#define DRCOMPILER_TRANSFORMS_DATARECOMPUTATION_STRATEGIES_FULLREMAT_H

#include "drcompiler/Transforms/DataRecomputation/Strategies/Strategy.h"

namespace dr::strategies {

/// Same-function rematerialization with load chaining. Walks the stored
/// value's operand tree and clones it at the load site. Loads inside the
/// tree are chained through SINGLE-provenance up to kMaxChainDepth.
class FullRemat final : public LoadStrategy {
public:
  llvm::StringRef name() const override { return "full-remat"; }
  Outcome tryApply(LoadCandidate &c, StrategyEnv &env) override;
};

} // namespace dr::strategies

#endif
