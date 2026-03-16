#ifndef DRCOMPILER_TRANSFORMS_RAISEMALLOCMEMREF_H
#define DRCOMPILER_TRANSFORMS_RAISEMALLOCMEMREF_H

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL_RAISEMALLOCTOMEMREFPASS
#include "drcompiler/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createRaiseMallocToMemRefPass();

} // namespace mlir

#endif // DRCOMPILER_TRANSFORMS_RAISEMALLOCMEMREF_H
