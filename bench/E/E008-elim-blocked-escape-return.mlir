// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-buffer-elim=true dr-summary=true})' 2>&1 | FileCheck %s --check-prefix=SUM
// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-buffer-elim=true dr-erase-eliminated-buffers=true})' | FileCheck %s --check-prefix=KEPT

// E008: Alloc returned from function → EscapesViaReturn → INFEASIBLE.

// SUM: buffer-elim {{.*}}: INFEASIBLE (escape=escapes-return

// KEPT-LABEL: func.func @escape_return
// KEPT:       memref.alloc
// KEPT:       memref.store
func.func @escape_return() -> memref<i32> {
  %a = memref.alloc() : memref<i32>
  %c = arith.constant 5 : i32
  memref.store %c, %a[] : memref<i32>
  return %a : memref<i32>
}
