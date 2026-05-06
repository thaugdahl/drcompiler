//===- ConstantGlobalFold.h - Strategy 0: constant-global peephole -------===//
//
// Pre-analysis peephole: replaces loads from `constant` memref globals
// (with splat-dense initializers) with arith.constant. Doesn't fit the
// per-load LoadStrategy interface — runs as a free function before the
// reaching-stores analysis kicks in.
//
//===----------------------------------------------------------------------===//
#ifndef DRCOMPILER_TRANSFORMS_DATARECOMPUTATION_STRATEGIES_CONSTANTGLOBALFOLD_H
#define DRCOMPILER_TRANSFORMS_DATARECOMPUTATION_STRATEGIES_CONSTANTGLOBALFOLD_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

namespace dr::strategies {

/// Walk the module and replace splat-dense constant-global loads with
/// arith.constant. Erases the folded loads.
void runConstantGlobalFold(mlir::ModuleOp moduleOp,
                           mlir::SymbolTableCollection &symTab,
                           bool emitDiagnostics, bool summaryEnabled = false);

} // namespace dr::strategies

#endif
