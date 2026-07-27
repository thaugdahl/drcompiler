// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-buffer-elim=true dr-summary=true})' 2>&1 | FileCheck %s --check-prefix=SUM
// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-buffer-elim=true dr-erase-eliminated-buffers=true})' | FileCheck %s --check-prefix=KEPT

// E010: Two stores to the same scalar buffer under a conditional → the load
// is MULTI-provenance. Buffer-elim must report INFEASIBLE (multi>=1).

// SUM: buffer-elim {{.*}}: INFEASIBLE {{.*}}multi=1

// KEPT-LABEL: func.func @multi_blocks
// KEPT:       memref.alloca
func.func @multi_blocks(%cond: i1) -> i32 {
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
