#ifndef DRCOMPILER_TRANSFORMS_SCALARREDUCTIONDEMOTE_H
#define DRCOMPILER_TRANSFORMS_SCALARREDUCTIONDEMOTE_H

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL_DRSCALARREDUCTIONDEMOTEPASS
#include "drcompiler/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createDrScalarReductionDemotePass();

} // namespace mlir

#endif // DRCOMPILER_TRANSFORMS_SCALARREDUCTIONDEMOTE_H
