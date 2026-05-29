#ifndef DRCOMPILER_TRANSFORMS_AFFINELOOPFUSION_H
#define DRCOMPILER_TRANSFORMS_AFFINELOOPFUSION_H

// LLVM trunk re-orgs the Affine Passes.h location; support both layouts so
// drcompiler builds against either Marco's bundled LLVM (top-level) or an
// installed 22.1.x (under Transforms/).
#if __has_include("mlir/Dialect/Affine/Passes.h")
#include "mlir/Dialect/Affine/Passes.h"
#else
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#endif
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
using affine::FusionMode;
#define GEN_PASS_DECL_DRAFFINELOOPFUSIONPASS
#include "drcompiler/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createDrAffineLoopFusionPass();
std::unique_ptr<Pass> createDrAffineLoopFusionPass(
    unsigned fastMemorySpace, uint64_t localBufSizeThreshold,
    bool maximalFusion, enum affine::FusionMode affineFusionMode);
} // namespace mlir

#endif // DRCOMPILER_TRANSFORMS_AFFINELOOPFUSION_H
