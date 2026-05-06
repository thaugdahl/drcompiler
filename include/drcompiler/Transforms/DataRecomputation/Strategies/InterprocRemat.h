//===- InterprocRemat.h - Strategy 3: callee-side interproc remat --------===//
#ifndef DRCOMPILER_TRANSFORMS_DATARECOMPUTATION_STRATEGIES_INTERPROCREMAT_H
#define DRCOMPILER_TRANSFORMS_DATARECOMPUTATION_STRATEGIES_INTERPROCREMAT_H

#include "drcompiler/Transforms/DataRecomputation/Strategies/Strategy.h"

namespace dr::strategies {

/// Cross-function rematerialization via callee-arg → caller-operand
/// mapping. Runs when load and store are in different functions and the
/// InterproceduralOriginMap records exactly one origin for the store.
/// On rejection, the partial-remat fallback (interproc mode) runs
/// per-origin inside the same loop.
class InterprocRemat final : public LoadStrategy {
public:
  llvm::StringRef name() const override { return "interproc-remat"; }
  Outcome tryApply(LoadCandidate &c, StrategyEnv &env) override;
};

} // namespace dr::strategies

#endif
