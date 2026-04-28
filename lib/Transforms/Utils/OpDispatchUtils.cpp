//===- OpDispatchUtils.cpp ------------------------------------------------===//

#include "drcompiler/Transforms/Utils/OpDispatchUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace drcompiler {

bool isAnyLoadOp(mlir::Operation *op) {
  return mlir::isa_and_nonnull<mlir::memref::LoadOp, mlir::affine::AffineLoadOp,
                               mlir::LLVM::LoadOp>(op);
}

bool isAnyStoreOp(mlir::Operation *op) {
  return mlir::isa_and_nonnull<mlir::memref::StoreOp,
                               mlir::affine::AffineStoreOp,
                               mlir::LLVM::StoreOp>(op);
}

mlir::Value getLoadStoreMemref(mlir::Operation *op) {
  if (auto o = mlir::dyn_cast_or_null<mlir::memref::LoadOp>(op))
    return o.getMemRef();
  if (auto o = mlir::dyn_cast_or_null<mlir::memref::StoreOp>(op))
    return o.getMemRef();
  if (auto o = mlir::dyn_cast_or_null<mlir::affine::AffineLoadOp>(op))
    return o.getMemRef();
  if (auto o = mlir::dyn_cast_or_null<mlir::affine::AffineStoreOp>(op))
    return o.getMemRef();
  if (auto o = mlir::dyn_cast_or_null<mlir::LLVM::LoadOp>(op))
    return o.getAddr();
  if (auto o = mlir::dyn_cast_or_null<mlir::LLVM::StoreOp>(op))
    return o.getAddr();
  return {};
}

mlir::Value getStoredValue(mlir::Operation *op) {
  if (auto o = mlir::dyn_cast_or_null<mlir::memref::StoreOp>(op))
    return o.getValueToStore();
  if (auto o = mlir::dyn_cast_or_null<mlir::affine::AffineStoreOp>(op))
    return o.getValueToStore();
  if (auto o = mlir::dyn_cast_or_null<mlir::LLVM::StoreOp>(op))
    return o.getValue();
  return {};
}

mlir::ValueRange getLoadStoreIndices(mlir::Operation *op) {
  if (auto o = mlir::dyn_cast_or_null<mlir::memref::LoadOp>(op))
    return o.getIndices();
  if (auto o = mlir::dyn_cast_or_null<mlir::memref::StoreOp>(op))
    return o.getIndices();
  if (auto o = mlir::dyn_cast_or_null<mlir::affine::AffineLoadOp>(op))
    return o.getIndices();
  if (auto o = mlir::dyn_cast_or_null<mlir::affine::AffineStoreOp>(op))
    return o.getIndices();
  return {};
}

mlir::Type getLoadStoreElementType(mlir::Operation *op) {
  mlir::Value memref = getLoadStoreMemref(op);
  if (!memref)
    return {};
  if (auto mt = mlir::dyn_cast<mlir::MemRefType>(memref.getType()))
    return mt.getElementType();
  // For LLVM pointer ops, derive from the loaded result / stored value.
  if (auto load = mlir::dyn_cast<mlir::LLVM::LoadOp>(op))
    return load.getRes().getType();
  if (auto store = mlir::dyn_cast<mlir::LLVM::StoreOp>(op))
    return store.getValue().getType();
  return {};
}

} // namespace drcompiler
