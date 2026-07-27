#ifndef DRCOMPILER_TRANSFORMS_RAISESCFTOAFFINE_H
#define DRCOMPILER_TRANSFORMS_RAISESCFTOAFFINE_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
#define GEN_PASS_DECL_DRRAISESCFTOAFFINEPASS
#include "drcompiler/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createDrRaiseSCFToAffinePass();
} // namespace mlir

#endif // DRCOMPILER_TRANSFORMS_RAISESCFTOAFFINE_H
