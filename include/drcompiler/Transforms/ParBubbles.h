#ifndef DRCOMPILER_TRANSFORMS_PARBUBBLES_H
#define DRCOMPILER_TRANSFORMS_PARBUBBLES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL_DRPARBUBBLESPASS
#include "drcompiler/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createDrParBubblesPass();

} // namespace mlir

#endif // DRCOMPILER_TRANSFORMS_PARBUBBLES_H
