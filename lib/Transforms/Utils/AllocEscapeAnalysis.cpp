//===- AllocEscapeAnalysis.cpp --------------------------------------------===//

#include "drcompiler/Transforms/Utils/AllocEscapeAnalysis.h"
#include "drcompiler/Transforms/Utils/MemrefBaseAnalysis.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace drcompiler {

namespace {

/// True if `v` is the memref/pointer operand (not the value-being-stored) of
/// any store op. We only call this when we know `op` is a store.
bool isMemrefOperandOfStore(mlir::Operation *op, mlir::Value v) {
  if (auto s = mlir::dyn_cast<mlir::memref::StoreOp>(op))
    return s.getMemRef() == v;
  if (auto s = mlir::dyn_cast<mlir::affine::AffineStoreOp>(op))
    return s.getMemRef() == v;
  if (auto s = mlir::dyn_cast<mlir::LLVM::StoreOp>(op))
    return s.getAddr() == v;
  if (auto s = mlir::dyn_cast<mlir::vector::StoreOp>(op))
    return s.getBase() == v;
  if (auto s = mlir::dyn_cast<mlir::vector::TransferWriteOp>(op))
    return s.getBase() == v;
  return false;
}

bool isLoadLike(mlir::Operation *op) {
  return mlir::isa<mlir::memref::LoadOp, mlir::affine::AffineLoadOp,
                   mlir::LLVM::LoadOp, mlir::vector::LoadOp,
                   mlir::vector::TransferReadOp>(op);
}

bool isStoreLike(mlir::Operation *op) {
  return mlir::isa<mlir::memref::StoreOp, mlir::affine::AffineStoreOp,
                   mlir::LLVM::StoreOp, mlir::vector::StoreOp,
                   mlir::vector::TransferWriteOp>(op);
}

bool isDeallocLike(mlir::Operation *op) {
  // memref.dealloc and llvm.call @free are both safe sinks. We do not import
  // the LLVM dialect's call form here — callers already see free() as a
  // CallOpInterface and the oracle handles it.
  return mlir::isa<mlir::memref::DeallocOp>(op);
}

bool isReturnLike(mlir::Operation *op) {
  return op->hasTrait<mlir::OpTrait::ReturnLike>() ||
         mlir::isa<mlir::RegionBranchTerminatorOpInterface>(op);
}

} // namespace

EscapeResult analyzeAllocEscape(mlir::Operation *allocOp,
                                CallEscapeOracle callOracle) {
  EscapeResult res;

  llvm::SmallVector<mlir::Value, 8> worklist;
  llvm::SmallDenseSet<mlir::Value> seen;

  auto pushIfTracked = [&](mlir::Value v) {
    if (!v) return;
    auto t = v.getType();
    if (!mlir::isa<mlir::MemRefType, mlir::LLVM::LLVMPointerType,
                   mlir::UnrankedMemRefType>(t))
      return;
    if (seen.insert(v).second)
      worklist.push_back(v);
  };

  for (mlir::Value r : allocOp->getResults())
    pushIfTracked(r);

  while (!worklist.empty()) {
    mlir::Value cur = worklist.pop_back_val();

    for (mlir::OpOperand &use : cur.getUses()) {
      mlir::Operation *user = use.getOwner();

      // View/cast/GEP/polygeist.memref2pointer → recurse on every tracked
      // result. isViewLikeOp covers all five.
      if (isViewLikeOp(user)) {
        for (mlir::Value r : user->getResults())
          pushIfTracked(r);
        continue;
      }

      // Safe sinks: load + dealloc + return-via-dealloc.
      if (isLoadLike(user) || isDeallocLike(user))
        continue;

      // Store: ok if the memref operand is being stored INTO; escape if the
      // pointer is the value being stored ELSEWHERE.
      if (isStoreLike(user)) {
        if (isMemrefOperandOfStore(user, cur))
          continue;
        res.kind = EscapeKind::EscapesAsPtrValue;
        res.escapeOp = user;
        return res;
      }

      // Return / yield: escape only when the tracked value is among the
      // operands flowing out. ReturnLike includes func.return + llvm.return.
      if (isReturnLike(user)) {
        res.kind = EscapeKind::EscapesViaReturn;
        res.escapeOp = user;
        return res;
      }

      // Call: defer to oracle. Unset oracle => conservative escape.
      if (auto call = mlir::dyn_cast<mlir::CallOpInterface>(user)) {
        unsigned argIdx = use.getOperandNumber();
        bool callEscapes = callOracle ? callOracle(call, argIdx) : true;
        if (callEscapes) {
          res.kind = EscapeKind::EscapesToCall;
          res.escapeOp = user;
          return res;
        }
        continue;
      }

      // Pointer-to-integer cast → escape.
      if (mlir::isa<mlir::LLVM::PtrToIntOp>(user)) {
        res.kind = EscapeKind::EscapesAsPtrValue;
        res.escapeOp = user;
        return res;
      }

      // memref.copy is symmetric: both operands are memref bases, treat as
      // load+store (no escape).
      if (mlir::isa<mlir::memref::CopyOp>(user))
        continue;

      // Anything else → unknown consumer.
      res.kind = EscapeKind::EscapesUnknown;
      res.escapeOp = user;
      return res;
    }
  }

  return res;
}

} // namespace drcompiler
