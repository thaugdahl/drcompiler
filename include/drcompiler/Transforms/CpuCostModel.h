#ifndef DRCOMPILER_TRANSFORMS_CPUCOSTMODEL_H
#define DRCOMPILER_TRANSFORMS_CPUCOSTMODEL_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace drcompiler {

/// Optional cache hierarchy parameters parsed from a cost-model JSON's
/// `cache` object.  Each field is std::nullopt unless present in the file.
/// Consumers should fall back to CLI flags / built-in defaults for any
/// field left unset.
struct CpuCacheJsonParams {
  std::optional<unsigned> l1Size;
  std::optional<unsigned> l2Size;
  std::optional<unsigned> l3Size;
  std::optional<unsigned> l1Latency;
  std::optional<unsigned> l2Latency;
  std::optional<unsigned> l3Latency;
  std::optional<unsigned> memLatency;
};

/// A table mapping MLIR operation names (e.g. "arith.addi") to estimated
/// cycle costs.  Loaded from a JSON file or populated with built-in defaults.
class CpuCostModel {
public:
  /// Load a cost model from a JSON file.  On success, returns the model.
  /// On failure, logs a warning and returns a model with built-in defaults.
  static CpuCostModel loadFromFile(llvm::StringRef path);

  /// Return a model with the built-in default cost table.
  static CpuCostModel getDefault();

  /// Look up the cycle cost of a single operation.
  unsigned opCost(mlir::Operation *op) const;

  /// Whether this model was loaded from a file (vs. built-in defaults).
  bool isFromFile() const { return fromFile; }

  /// Optional cache hierarchy parameters parsed from the JSON file.  Fields
  /// are nullopt unless they were present and well-formed.
  const CpuCacheJsonParams &cacheParams() const { return cache; }

private:
  llvm::StringMap<unsigned> table;
  unsigned defaultCost = 5;
  bool fromFile = false;
  CpuCacheJsonParams cache;

  void populateDefaults();
};

} // namespace drcompiler

#endif // DRCOMPILER_TRANSFORMS_CPUCOSTMODEL_H
