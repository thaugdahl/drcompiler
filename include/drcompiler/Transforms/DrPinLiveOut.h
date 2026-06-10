#ifndef DRCOMPILER_TRANSFORMS_DRPINLIVEOUT_H
#define DRCOMPILER_TRANSFORMS_DRPINLIVEOUT_H

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL_DRPINLIVEOUTPASS
#include "drcompiler/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createDrPinLiveOutPass();

} // namespace mlir

#endif // DRCOMPILER_TRANSFORMS_DRPINLIVEOUT_H
