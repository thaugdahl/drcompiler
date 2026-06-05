#ifndef DRCOMPILER_TRANSFORMS_AFFINEREGISTERBLOCK_H
#define DRCOMPILER_TRANSFORMS_AFFINEREGISTERBLOCK_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
#define GEN_PASS_DECL_AFFINEREGISTERBLOCKPASS
#include "drcompiler/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createAffineRegisterBlockPass();
} // namespace mlir

#endif // DRCOMPILER_TRANSFORMS_AFFINEREGISTERBLOCK_H
