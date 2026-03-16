#ifndef DRCOMPILER_TRANSFORMS_MEMORYFISSION_H
#define DRCOMPILER_TRANSFORMS_MEMORYFISSION_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
#define GEN_PASS_DECL_MEMORYFISSIONPASS
#include "drcompiler/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createMemoryFissionPass();
} // namespace mlir

#endif // DRCOMPILER_TRANSFORMS_MEMORYFISSION_H
