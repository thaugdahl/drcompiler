//===- MemrefBaseAnalysis.h - Memref alloc-root tracing --------*- C++ -*-===//
//
// Trace a memref/pointer SSA value back to its allocation roots through
// view-like ops (memref.subview, memref.reinterpret_cast, memref.view,
// memref.cast), llvm.getelementptr, and polygeist.memref2pointer.
//
// Companion to AllocationRoots (a Value -> alloc Op map). Extracted so
// passes other than DataRecomputation can reuse the traversal.
//
//===----------------------------------------------------------------------===//

#ifndef DRCOMPILER_TRANSFORMS_UTILS_MEMREFBASEANALYSIS_H
#define DRCOMPILER_TRANSFORMS_UTILS_MEMREFBASEANALYSIS_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace drcompiler {

/// Append every base memref reachable from `v` by walking view-like ops
/// to `bases`. A base is reached when:
///   1. the value is in `allocRootFor`,
///   2. the value is a BlockArgument (the function's entry inputs), or
///   3. there is no further view-like producer to peel.
///
/// `allocRootFor` is the value-to-alloc-op map maintained by the pass
/// (see drcompiler::dr::AllocationRoots).
void collectBaseMemrefs(
    mlir::Value v,
    const llvm::DenseMap<mlir::Value, mlir::Operation *> &allocRootFor,
    llvm::SmallVectorImpl<mlir::Value> &bases);

/// Returns true if `op` is a view-like op recognised by collectBaseMemrefs:
/// memref.subview, memref.reinterpret_cast, memref.view, memref.cast,
/// llvm.getelementptr, or polygeist.memref2pointer.
bool isViewLikeOp(mlir::Operation *op);

} // namespace drcompiler

#endif // DRCOMPILER_TRANSFORMS_UTILS_MEMREFBASEANALYSIS_H
