#ifndef DRCOMPILER_TRANSFORMS_CONVERTPARTOOMP_H
#define DRCOMPILER_TRANSFORMS_CONVERTPARTOOMP_H

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL_CONVERTPARTOOMPPASS
#include "drcompiler/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createConvertParToOMPPass();

} // namespace mlir

#endif // DRCOMPILER_TRANSFORMS_CONVERTPARTOOMP_H
