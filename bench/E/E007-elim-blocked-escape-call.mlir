// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-buffer-elim=true dr-summary=true})' 2>&1 | FileCheck %s --check-prefix=SUM
// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-buffer-elim=true dr-erase-eliminated-buffers=true})' | FileCheck %s --check-prefix=KEPT

// E007: Alloc passed to external function (no body) → EscapesToCall → INFEASIBLE.

// SUM: buffer-elim {{.*}}: INFEASIBLE (escape=escapes-call

// KEPT-LABEL: func.func @escape_call
// KEPT:       memref.alloca
// KEPT:       memref.store
func.func private @extern_sink(memref<i32>) -> ()

func.func @escape_call() {
  %a = memref.alloca() : memref<i32>
  %c = arith.constant 7 : i32
  memref.store %c, %a[] : memref<i32>
  func.call @extern_sink(%a) : (memref<i32>) -> ()
  return
}
