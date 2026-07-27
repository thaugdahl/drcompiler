// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-buffer-elim=true dr-summary=true})' 2>&1 | FileCheck %s --check-prefix=SUM
// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-buffer-elim=true dr-erase-eliminated-buffers=true})' | FileCheck %s --check-prefix=ERASED

// G013: DR replaces 3 of 4 SINGLE loads via remat. The 4th is also SINGLE.
// Buffer elim sees all loads are SINGLE and no escapes, so the buffer
// is FEASIBLE for elimination. With erase, alloca and stores vanish.

func.func @elim_after_remat() -> i32 {
  %a = memref.alloca() : memref<4xi32>
  %c = arith.constant 42 : i32
  affine.store %c, %a[0] : memref<4xi32>
  affine.store %c, %a[1] : memref<4xi32>
  affine.store %c, %a[2] : memref<4xi32>
  affine.store %c, %a[3] : memref<4xi32>
  %v0 = affine.load %a[0] : memref<4xi32>
  %v1 = affine.load %a[1] : memref<4xi32>
  %v2 = affine.load %a[2] : memref<4xi32>
  %v3 = affine.load %a[3] : memref<4xi32>
  %s0 = arith.addi %v0, %v1 : i32
  %s1 = arith.addi %s0, %v2 : i32
  %s2 = arith.addi %s1, %v3 : i32
  return %s2 : i32
}

// Summary: all loads SINGLE, no escape.
// SUM: buffer-elim {{.*}}: FEASIBLE

// Erased: alloca and stores removed, loads replaced by constant.
// ERASED-LABEL: func.func @elim_after_remat
// ERASED-NOT:   memref.alloca
// ERASED-NOT:   memref.store
// ERASED-NOT:   affine.store
// ERASED:       arith.constant 42
// ERASED:       return
