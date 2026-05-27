//===- AllocEscapeAnalysis.h - Whole-allocation escape analysis -*- C++ -*-===//
//
// Decide whether every transitive use of an allocation root can be replaced
// by recomputation. Reports the first user that prevents elimination.
//
//===----------------------------------------------------------------------===//

#ifndef DRCOMPILER_TRANSFORMS_UTILS_ALLOCESCAPEANALYSIS_H
#define DRCOMPILER_TRANSFORMS_UTILS_ALLOCESCAPEANALYSIS_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace drcompiler {

enum class EscapeKind {
  NoEscape,           // every use is load/store/dealloc/view/well-behaved call
  EscapesToCall,      // passed to a call whose callee escapes the arg further
  EscapesViaReturn,   // alloc value flows to a return / yield
  EscapesAsPtrValue,  // stored as a pointer value or cast to integer
  EscapesUnknown,     // unknown consumer (op kind not on the whitelist)
};

struct EscapeResult {
  EscapeKind kind = EscapeKind::NoEscape;
  mlir::Operation *escapeOp = nullptr; // first offending user, for diagnostics
};

/// Oracle: given a call site and the arg index that receives the tracked
/// value, return true if the callee escapes that argument further (i.e., the
/// pointer flows out of the callee). False means the callee only reads/writes
/// through it directly. An unset oracle treats every call as an escape.
using CallEscapeOracle =
    llvm::function_ref<bool(mlir::CallOpInterface, unsigned)>;

/// Walk every transitive use of every memref/LLVM-ptr result of `allocOp`,
/// chasing view-like producers (memref.subview, reinterpret_cast, view,
/// cast, llvm.getelementptr, polygeist.memref2pointer). Reports NoEscape
/// when every leaf use is a known load/store/dealloc/copy/return-free op
/// or a call the oracle approves.
EscapeResult analyzeAllocEscape(mlir::Operation *allocOp,
                                CallEscapeOracle callOracle = {});

} // namespace drcompiler

#endif // DRCOMPILER_TRANSFORMS_UTILS_ALLOCESCAPEANALYSIS_H
