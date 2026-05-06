//===- PartialRemat.h - Strategy 2b: leaf-load partial remat -------------===//
#ifndef DRCOMPILER_TRANSFORMS_DATARECOMPUTATION_STRATEGIES_PARTIALREMAT_H
#define DRCOMPILER_TRANSFORMS_DATARECOMPUTATION_STRATEGIES_PARTIALREMAT_H

#include "drcompiler/Transforms/DataRecomputation/Strategies/Strategy.h"
#include "mlir/IR/IRMapping.h"

namespace dr::strategies {

/// Partial rematerialization. Permits non-chainable leaf loads to be
/// re-emitted as loads at the remat site (instead of rejecting the
/// remat). Two contexts:
///
///   * Intraprocedural — argMapping == nullptr, strict cost gate
///     (alu+leaf < loadLat).
///   * Interprocedural — argMapping maps callee entry args to caller
///     operands; cost gate via decideBufferStrategy.
class PartialRemat final : public LoadStrategy {
public:
  enum class Mode { Intraprocedural, Interprocedural };

  PartialRemat(Mode mode, const mlir::IRMapping *argMapping = nullptr)
      : mode(mode), argMapping(argMapping) {}

  llvm::StringRef name() const override { return "partial-remat"; }
  Outcome tryApply(LoadCandidate &c, StrategyEnv &env) override;

private:
  Mode mode;
  const mlir::IRMapping *argMapping;
};

} // namespace dr::strategies

#endif
