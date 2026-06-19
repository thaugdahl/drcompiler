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
  size_t nDyn = 0;
  for (int64_t ub : getUpperBounds())
    if (ShapedType::isDynamic(ub))
      ++nDyn;
  if (getDynamicUpperBounds().size() != nDyn)
    return emitOpError("expected ")
           << nDyn << " dynamic upper-bound operand(s) (one per kDynamic entry "
              "in upperBounds), got "
           << getDynamicUpperBounds().size();
  if (getResults().size() != getInitVals().size())
    return emitOpError("expected ")
           << getInitVals().size() << " result(s) (one per init value), got "
           << getResults().size();
  // Terminator: par.reduce iff there are results; else par.yield (the generic
  // SingleBlock terminator).  Reduce contributions/results must line up.
  Operation *term = getBody()->getTerminator();
  if (!getResults().empty()) {
    auto red = dyn_cast<ReduceOp>(term);
    if (!red)
      return emitOpError("a reducing par.forall must end in par.reduce");
    if (red.getContributions().size() != getResults().size())
      return emitOpError("par.reduce contributes ")
             << red.getContributions().size() << " value(s) but forall has "
             << getResults().size() << " result(s)";
    for (auto [r, c] : llvm::zip(getResultTypes(), red.getContributions()))
      if (r != c.getType())
        return emitOpError("par.reduce contribution type does not match the "
                           "forall result type");
  } else if (isa<ReduceOp>(term)) {
    return emitOpError("par.reduce present but the forall has no results");
  }
  return success();
}

LogicalResult ReduceOp::verify() {
  if (getKinds().size() != getContributions().size())
    return emitOpError("expected one kind per contribution (")
           << getContributions().size() << "), got " << getKinds().size();
  for (int64_t k : getKinds())
    if (k < 0 || k > 7)
      return emitOpError("reduction kind out of range [0,7]: ") << k;
  return success();
}
