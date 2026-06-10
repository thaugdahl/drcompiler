#ifndef DRCOMPILER_TRANSFORMS_TESTREUSEANALYSIS_H
#define DRCOMPILER_TRANSFORMS_TESTREUSEANALYSIS_H

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL_DRTESTREUSEANALYSISPASS
#include "drcompiler/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createDrTestReuseAnalysisPass();

} // namespace mlir

#endif // DRCOMPILER_TRANSFORMS_TESTREUSEANALYSIS_H
