#ifndef DRCOMPILER_TRANSFORMS_CPUCOSTMODEL_H
#define DRCOMPILER_TRANSFORMS_CPUCOSTMODEL_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace drcompiler {

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

private:
  llvm::StringMap<unsigned> table;
  unsigned defaultCost = 5;
  bool fromFile = false;

  void populateDefaults();
};

} // namespace drcompiler

#endif // DRCOMPILER_TRANSFORMS_CPUCOSTMODEL_H
