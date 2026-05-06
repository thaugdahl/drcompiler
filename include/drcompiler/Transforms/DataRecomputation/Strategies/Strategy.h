//===- Strategy.h - Per-load rematerialization strategy interface --------===//
//
//===----------------------------------------------------------------------===//
//
// A LoadStrategy decides whether and how to replace a single SINGLE-
// provenance memref.load with a recomputed value. Strategies fire in
// pipeline order, first-match-wins. The pipeline driver lives in
// runOnOperation; individual strategies live one-per-file.
//
//===----------------------------------------------------------------------===//

#ifndef DRCOMPILER_TRANSFORMS_DATARECOMPUTATION_STRATEGIES_STRATEGY_H
#define DRCOMPILER_TRANSFORMS_DATARECOMPUTATION_STRATEGIES_STRATEGY_H

#include "drcompiler/Transforms/CpuCostModel.h"
#include "drcompiler/Transforms/DataRecomputation/AnalysisState.h"
#include "drcompiler/Transforms/DataRecomputation/CacheCostModel.h"
#include "drcompiler/Transforms/DataRecomputation/RematKernel.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/raw_ostream.h"

namespace dr::strategies {

/// One SINGLE-provenance load identified for rewriting.
struct LoadCandidate {
  mlir::Operation *loadOp;
  mlir::Operation *storeOp;       // unique reaching store
  mlir::Value      storedValue;   // value passed to storeOp
};

/// Shared state threaded through every per-load strategy. Every reference
/// here outlives the strategy invocation — owned by runOnOperation.
struct StrategyEnv {
  mlir::DominanceInfo &dom;
  LoadProvenanceMap &loadProv;
  InterproceduralOriginMap &interprocOrigins;
  llvm::DenseMap<mlir::Value, mlir::Operation *> &allocRootFor;
  const RootWriteMap &rootWrites;
  const CacheParams &cache;
  const drcompiler::CpuCostModel &costModel;
  bool partialRematEnabled;
  unsigned partialMaxLeaves;
  bool emitDiagnostics;
  bool summaryEnabled;
};

/// Print one DRSUM line for a per-load remat decision when --dr-summary is on.
/// Format: `DRSUM: load <loc>: <msg>` (e.g. "direct-forward ACCEPT").
inline void summaryLog(const StrategyEnv &env, mlir::Operation *op,
                       llvm::StringRef msg) {
  if (!env.summaryEnabled) return;
  llvm::errs() << "DRSUM: load ";
  op->getLoc().print(llvm::errs());
  llvm::errs() << ": " << msg << "\n";
}

/// Result of a strategy attempt.
enum class Outcome {
  /// Strategy fired and rewrote the load. Pipeline stops for this candidate.
  Accepted,
  /// Strategy did not fire — pipeline continues to the next strategy.
  NotApplicable,
};

/// Abstract base for per-load strategies.
class LoadStrategy {
public:
  virtual ~LoadStrategy() = default;
  virtual llvm::StringRef name() const = 0;
  virtual Outcome tryApply(LoadCandidate &c, StrategyEnv &env) = 0;
};

} // namespace dr::strategies

#endif // DRCOMPILER_TRANSFORMS_DATARECOMPUTATION_STRATEGIES_STRATEGY_H
