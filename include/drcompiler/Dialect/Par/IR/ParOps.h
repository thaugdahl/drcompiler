#ifndef DRCOMPILER_DIALECT_PAR_IR_PAROPS_H
#define DRCOMPILER_DIALECT_PAR_IR_PAROPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "drcompiler/Dialect/Par/IR/ParOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "drcompiler/Dialect/Par/IR/ParOps.h.inc"

#endif // DRCOMPILER_DIALECT_PAR_IR_PAROPS_H
