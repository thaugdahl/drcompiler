// RUN: dr-opt %s --allow-unregistered-dialect --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-buffer-elim=true dr-summary=true})' 2>&1 | FileCheck %s --check-prefix=SUM
// RUN: dr-opt %s --allow-unregistered-dialect --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-buffer-elim=true dr-erase-eliminated-buffers=true})' | FileCheck %s --check-prefix=KEPT

// E012: Alloc used by an unrecognized op (not load/store/dealloc/view/call/
// return/ptrtoint) → EscapesUnknown → INFEASIBLE.

// SUM: buffer-elim {{.*}}: INFEASIBLE (escape=escapes-unknown

// KEPT-LABEL: func.func @unknown_op
// KEPT:       memref.alloca
// KEPT:       memref.store
func.func @unknown_op() {
  %a = memref.alloca() : memref<i32>
  %c = arith.constant 3 : i32
  memref.store %c, %a[] : memref<i32>
  "custom.mystery"(%a) : (memref<i32>) -> ()
  return
}
