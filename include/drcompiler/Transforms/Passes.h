#ifndef DRCOMPILER_TRANSFORMS_PASSES_H
#define DRCOMPILER_TRANSFORMS_PASSES_H

#include "drcompiler/Transforms/DataRecomputation.h"

namespace mlir {

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "drcompiler/Transforms/Passes.h.inc"
} // namespace mlir

#endif // DRCOMPILER_TRANSFORMS_PASSES_H
