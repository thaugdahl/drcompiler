// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-buffer-elim=true dr-summary=true})' 2>&1 | FileCheck %s --check-prefix=SUM
// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-buffer-elim=true dr-erase-eliminated-buffers=true})' | FileCheck %s --check-prefix=KEPT

// Two stores to the same buffer at the same index → the load is MULTI.
// Buffer-elim must report INFEASIBLE (multi>=1) and refuse to erase.

// SUM: buffer-elim {{.*}}: INFEASIBLE {{.*}}multi=1

// KEPT-LABEL: func.func @multi
// KEPT:       memref.alloca
func.func @multi(%cond: i1) -> i32 {
  %a = memref.alloca() : memref<i32>
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  scf.if %cond {
    memref.store %c0, %a[] : memref<i32>
  } else {
    memref.store %c1, %a[] : memref<i32>
  }
  %v = memref.load %a[] : memref<i32>
  return %v : i32
}
