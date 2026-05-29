#ifndef DRCOMPILER_TRANSFORMS_PRINTARCHHANDLER_H
#define DRCOMPILER_TRANSFORMS_PRINTARCHHANDLER_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
#define GEN_PASS_DECL_PRINTARCHHANDLERPASS
#include "drcompiler/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createPrintArchHandlerPass();
} // namespace mlir

#endif // DRCOMPILER_TRANSFORMS_PRINTARCHHANDLER_H
