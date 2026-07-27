// RUN: dr-opt %s --allow-unregistered-dialect --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-buffer-elim=true dr-summary=true})' 2>&1 | FileCheck %s --check-prefix=SUM
// RUN: dr-opt %s --allow-unregistered-dialect --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-buffer-elim=true dr-erase-eliminated-buffers=true})' | FileCheck %s --check-prefix=ERASED

// E003: 4 stores to different indices, 4 matching loads → all SINGLE → FEASIBLE.

// SUM: buffer-elim {{.*}}: FEASIBLE (escape=no-escape, loads=4

// ERASED-LABEL: func.func @multi_store
// ERASED-NOT:   memref.alloca
// ERASED-NOT:   affine.store
// ERASED:       %[[C0:.+]] = arith.constant 10 : i32
// ERASED:       %[[C1:.+]] = arith.constant 20 : i32
// ERASED:       %[[C2:.+]] = arith.constant 30 : i32
// ERASED:       %[[C3:.+]] = arith.constant 40 : i32
// ERASED:       "use.consume"(%[[C0]], %[[C1]], %[[C2]], %[[C3]])
func.func @multi_store() {
  %a = memref.alloca() : memref<4xi32>
  %c0 = arith.constant 10 : i32
  %c1 = arith.constant 20 : i32
  %c2 = arith.constant 30 : i32
  %c3 = arith.constant 40 : i32
  affine.store %c0, %a[0] : memref<4xi32>
  affine.store %c1, %a[1] : memref<4xi32>
  affine.store %c2, %a[2] : memref<4xi32>
  affine.store %c3, %a[3] : memref<4xi32>
  %v0 = affine.load %a[0] : memref<4xi32>
  %v1 = affine.load %a[1] : memref<4xi32>
  %v2 = affine.load %a[2] : memref<4xi32>
  %v3 = affine.load %a[3] : memref<4xi32>
  "use.consume"(%v0, %v1, %v2, %v3) : (i32, i32, i32, i32) -> ()
  return
}
