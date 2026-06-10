#ifndef DRCOMPILER_TRANSFORMS_AFFINESTENCILTIMETILE_H
#define DRCOMPILER_TRANSFORMS_AFFINESTENCILTIMETILE_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
#define GEN_PASS_DECL_DRAFFINESTENCILTIMETILEPASS
#include "drcompiler/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createDrAffineStencilTimeTilePass();
} // namespace mlir

#endif // DRCOMPILER_TRANSFORMS_AFFINESTENCILTIMETILE_H
