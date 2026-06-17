#ifndef DRCOMPILER_TRANSFORMS_MATHSTRENGTHREDUCE_H
#define DRCOMPILER_TRANSFORMS_MATHSTRENGTHREDUCE_H

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL_DRMATHSTRENGTHREDUCEPASS
#include "drcompiler/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createDrMathStrengthReducePass();

} // namespace mlir

#endif // DRCOMPILER_TRANSFORMS_MATHSTRENGTHREDUCE_H
