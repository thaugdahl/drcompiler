#ifndef DRCOMPILER_TRANSFORMS_AFFINELOOPTILE_H
#define DRCOMPILER_TRANSFORMS_AFFINELOOPTILE_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
#define GEN_PASS_DECL_DRAFFINELOOPTILEPASS
#include "drcompiler/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createDrAffineLoopTilePass();
} // namespace mlir

#endif // DRCOMPILER_TRANSFORMS_AFFINELOOPTILE_H
