#ifndef DRCOMPILER_TRANSFORMS_LOWERKRNLGLOBAL_H
#define DRCOMPILER_TRANSFORMS_LOWERKRNLGLOBAL_H

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL_LOWERKRNLGLOBALPASS
#include "drcompiler/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createLowerKrnlGlobalPass();

} // namespace mlir

#endif // DRCOMPILER_TRANSFORMS_LOWERKRNLGLOBAL_H
