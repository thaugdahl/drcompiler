#ifndef DRCOMPILER_TRANSFORMS_PASSES_H
#define DRCOMPILER_TRANSFORMS_PASSES_H

#include "drcompiler/Transforms/DataRecomputation.h"
#include "drcompiler/Transforms/MemoryFission.h"
#include "drcompiler/Transforms/RaiseMallocToMemRef.h"

namespace mlir {

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "drcompiler/Transforms/Passes.h.inc"
} // namespace mlir

#endif // DRCOMPILER_TRANSFORMS_PASSES_H
