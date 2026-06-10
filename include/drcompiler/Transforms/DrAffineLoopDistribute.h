#ifndef DRCOMPILER_TRANSFORMS_DRAFFINELOOPDISTRIBUTE_H
#define DRCOMPILER_TRANSFORMS_DRAFFINELOOPDISTRIBUTE_H

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL_DRAFFINELOOPDISTRIBUTEPASS
#include "drcompiler/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createDrAffineLoopDistributePass();

} // namespace mlir

#endif // DRCOMPILER_TRANSFORMS_DRAFFINELOOPDISTRIBUTE_H
