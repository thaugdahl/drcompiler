//===- MemrefBaseAnalysis.cpp ---------------------------------------------===//

#include "drcompiler/Transforms/Utils/MemrefBaseAnalysis.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace drcompiler {

bool isViewLikeOp(mlir::Operation *op) {
  if (!op)
    return false;
  if (mlir::isa<mlir::memref::SubViewOp, mlir::memref::ReinterpretCastOp,
                mlir::memref::ViewOp, mlir::memref::CastOp,
                mlir::LLVM::GEPOp>(op))
    return true;
  // polygeist.memref2pointer is from an out-of-tree dialect we don't link
  // against — match by registered op name.
  return op->getName().getStringRef() == "polygeist.memref2pointer";
}

void collectBaseMemrefs(
    mlir::Value v,
    const llvm::DenseMap<mlir::Value, mlir::Operation *> &allocRootFor,
    llvm::SmallVectorImpl<mlir::Value> &bases) {
  llvm::SmallVector<mlir::Value, 4> worklist{v};
  llvm::SmallDenseSet<mlir::Value> visited;

  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;

    if (allocRootFor.contains(current) ||
        mlir::isa<mlir::BlockArgument>(current)) {
      bases.push_back(current);
      continue;
    }

    if (mlir::Operation *def = current.getDefiningOp()) {
      if (isViewLikeOp(def)) {
        worklist.push_back(def->getOperand(0));
        continue;
      }
    }

    // Fallback: unknown producer — treat as a base of its own.
    bases.push_back(current);
  }
}

} // namespace drcompiler
