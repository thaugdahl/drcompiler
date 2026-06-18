//===- ParOps.cpp - `par` dialect implementation (M2) ------------------===//

#include "drcompiler/Dialect/Par/IR/ParOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::par;

#include "drcompiler/Dialect/Par/IR/ParOpsDialect.cpp.inc"

void ParDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "drcompiler/Dialect/Par/IR/ParOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "drcompiler/Dialect/Par/IR/ParOps.cpp.inc"

LogicalResult ForallOp::verify() {
  size_t n = getLowerBounds().size();
  if (getUpperBounds().size() != n || getSteps().size() != n)
    return emitOpError(
        "lowerBounds, upperBounds and steps must have equal length");
  if (getBody()->getNumArguments() != n)
    return emitOpError("expected ")
           << n << " index induction-variable block argument(s), got "
           << getBody()->getNumArguments();
  for (BlockArgument arg : getBody()->getArguments())
    if (!arg.getType().isIndex())
      return emitOpError("induction variables must be of 'index' type");
  return success();
}
