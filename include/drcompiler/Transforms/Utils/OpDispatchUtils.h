//===- OpDispatchUtils.h - Load/Store op dispatch helpers ------*- C++ -*-===//
//
// Centralizes the recurring "is this any load/store op, and what is its
// memref/value/indices?" dispatch across memref, affine, and LLVM dialects.
// Without these helpers, callers fan out into copy-pasted dyn_cast chains;
// see DataRecomputation.cpp for ~50 prior sites.
//
//===----------------------------------------------------------------------===//

#ifndef DRCOMPILER_TRANSFORMS_UTILS_OPDISPATCHUTILS_H
#define DRCOMPILER_TRANSFORMS_UTILS_OPDISPATCHUTILS_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

namespace drcompiler {

/// Returns true if `op` is a memref.load, affine.load, or llvm.load.
bool isAnyLoadOp(mlir::Operation *op);

/// Returns true if `op` is a memref.store, affine.store, or llvm.store.
bool isAnyStoreOp(mlir::Operation *op);

/// Returns the memref/pointer operand of a load or store op.
/// memref/affine ops use getMemRef(); LLVM ops use getAddr().
/// Returns null Value if `op` is not a recognized load or store.
mlir::Value getLoadStoreMemref(mlir::Operation *op);

/// Returns the value-being-stored for a store op.
/// memref/affine ops use getValueToStore(); LLVM uses getValue().
/// Returns null Value if `op` is not a recognized store.
mlir::Value getStoredValue(mlir::Operation *op);

/// Returns the index operands of a memref.load/store or affine.load/store.
/// LLVM load/store carry no index operands — returns empty range.
mlir::ValueRange getLoadStoreIndices(mlir::Operation *op);

/// Returns the element type of the memref/pointer operand of a load/store,
/// or null Type if the op is unrecognized or has no element type.
mlir::Type getLoadStoreElementType(mlir::Operation *op);

} // namespace drcompiler

#endif // DRCOMPILER_TRANSFORMS_UTILS_OPDISPATCHUTILS_H
