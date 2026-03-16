#ifndef DRCOMPILER_TRANSFORMS_DATARECOMPUTATION_H
#define DRCOMPILER_TRANSFORMS_DATARECOMPUTATION_H

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL_DATARECOMPUTATIONPASS
#include "drcompiler/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createDataRecomputationPass();

} // namespace mlir

#endif // DRCOMPILER_TRANSFORMS_DATARECOMPUTATION_H
