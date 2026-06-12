#ifndef DRCOMPILER_TRANSFORMS_SCALARREDUCTIONPROMOTE_H
#define DRCOMPILER_TRANSFORMS_SCALARREDUCTIONPROMOTE_H

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL_DRSCALARREDUCTIONPROMOTEPASS
#include "drcompiler/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createDrScalarReductionPromotePass();

} // namespace mlir

#endif // DRCOMPILER_TRANSFORMS_SCALARREDUCTIONPROMOTE_H
