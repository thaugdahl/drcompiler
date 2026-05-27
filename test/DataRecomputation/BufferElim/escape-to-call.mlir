// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-buffer-elim=true dr-summary=true})' 2>&1 | FileCheck %s --check-prefix=SUM
// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-buffer-elim=true dr-erase-eliminated-buffers=true})' | FileCheck %s --check-prefix=KEPT

// Buffer's pointer is passed to an external (declared-only) function.
// Escape analysis must mark it as EscapesToCall → INFEASIBLE → no erase.

// SUM: buffer-elim {{.*}}: INFEASIBLE (escape=escapes-call

// KEPT-LABEL: func.func @leaks
// KEPT:       memref.alloca
// KEPT:       memref.store
func.func private @sink(memref<i32>) -> ()

func.func @leaks() {
  %a = memref.alloca() : memref<i32>
  %c = arith.constant 7 : i32
  memref.store %c, %a[] : memref<i32>
  func.call @sink(%a) : (memref<i32>) -> ()
  return
}
