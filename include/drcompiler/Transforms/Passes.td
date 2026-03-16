#ifndef DRCOMPILER_TRANSFORMS_PASSES_TD
#define DRCOMPILER_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def DataRecomputationPass : Pass<"data-recomputation", "mlir::ModuleOp">
{
  let summary = "Applies data recomputation where likely to be profitable";

  let description = [{
    Load-store provenance analysis: for each memref.load, determines which
    memref.store operations could have been the last writer.

    Loads are classified into four categories:
      SINGLE  - exactly one known store (recomputation candidate)
      MULTI   - multiple possible stores
      LEAKED  - provenance includes an external/unknown write (nullptr)
      KILLED  - all reaching stores were killed (empty provenance set)

    When --dr-recompute is set, SINGLE loads are replaced by rematerializing
    the stored value at the load site.
  }];

  let options = [
    Option<"drTestDiagnostics", "dr-test-diagnostics", "bool", "false",
           "Enables test output">,
    Option<"drDotFile", "dr-dot-file", "std::string", /*default=*/"",
           "Output file path for load-provenance DOT graph">,
    Option<"drRecompute", "dr-recompute", "bool", "false",
           "Replace SINGLE loads with the reaching store's value">
  ];

  let dependentDialects = [
    "memref::MemRefDialect", "vector::VectorDialect",
    "affine::AffineDialect", "ptr::PtrDialect"
  ];

  let constructor = "mlir::createDataRecomputationPass()";
}

#endif
