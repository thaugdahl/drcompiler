#ifndef DRCOMPILER_TRANSFORMS_PRINTREGISTERPRESSURE_H
#define DRCOMPILER_TRANSFORMS_PRINTREGISTERPRESSURE_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
#define GEN_PASS_DECL_PRINTREGISTERPRESSUREPASS
#include "drcompiler/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createPrintRegisterPressurePass();
} // namespace mlir

#endif // DRCOMPILER_TRANSFORMS_PRINTREGISTERPRESSURE_H
