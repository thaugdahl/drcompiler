#ifndef DRCOMPILER_TRANSFORMS_CONVERTPARTOSCF_H
#define DRCOMPILER_TRANSFORMS_CONVERTPARTOSCF_H

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL_CONVERTPARTOSCFPASS
#include "drcompiler/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createConvertParToSCFPass();

} // namespace mlir

#endif // DRCOMPILER_TRANSFORMS_CONVERTPARTOSCF_H
