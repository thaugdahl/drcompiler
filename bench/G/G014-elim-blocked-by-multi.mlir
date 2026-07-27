// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-buffer-elim=true dr-summary=true})' 2>&1 | FileCheck %s --check-prefix=SUM
// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-buffer-elim=true dr-erase-eliminated-buffers=true})' | FileCheck %s --check-prefix=KEPT

// G014: 3 SINGLE loads + 1 MULTI load. The MULTI load blocks buffer
// elimination (elim requires all loads to be single-provenance).

func.func @elim_blocked(%cond: i1) -> i32 {
  %a = memref.alloca() : memref<i32>
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32

  // Conditional store creates MULTI provenance for subsequent load.
  scf.if %cond {
    memref.store %c0, %a[] : memref<i32>
  } else {
    memref.store %c1, %a[] : memref<i32>
  }

  // This load is MULTI (two possible store sources).
  %v_multi = memref.load %a[] : memref<i32>

  // Overwrite with a single store.
  memref.store %c1, %a[] : memref<i32>

  // These 3 loads are SINGLE provenance.
  %v1 = memref.load %a[] : memref<i32>
  %v2 = memref.load %a[] : memref<i32>
  %v3 = memref.load %a[] : memref<i32>

  %s0 = arith.addi %v_multi, %v1 : i32
  %s1 = arith.addi %s0, %v2 : i32
  %s2 = arith.addi %s1, %v3 : i32
  return %s2 : i32
}

// Buffer-elim blocked by multi.
// SUM: buffer-elim {{.*}}: INFEASIBLE {{.*}}multi=1

// Buffer survives because of MULTI load.
// KEPT-LABEL: func.func @elim_blocked
// KEPT:       memref.alloca
