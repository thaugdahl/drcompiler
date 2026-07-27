// RUN: dr-opt %s --allow-unregistered-dialect --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-cost-model=true dr-buffer-elim=true dr-summary=true})' 2>&1 | FileCheck %s --check-prefix=SUM

// E017: Buffer with many stores (16) but only 1 load. The store overhead
// in keepCost dominates: keepCost = loads*loadLat + stores*storeLat + ...
// The high store count tips the balance toward elimination.

// SUM: buffer-elim {{.*}}: FEASIBLE (escape=no-escape
func.func @many_stores() {
  %a = memref.alloca() : memref<16xi32>
  %c = arith.constant 7 : i32
  affine.store %c, %a[0]  : memref<16xi32>
  affine.store %c, %a[1]  : memref<16xi32>
  affine.store %c, %a[2]  : memref<16xi32>
  affine.store %c, %a[3]  : memref<16xi32>
  affine.store %c, %a[4]  : memref<16xi32>
  affine.store %c, %a[5]  : memref<16xi32>
  affine.store %c, %a[6]  : memref<16xi32>
  affine.store %c, %a[7]  : memref<16xi32>
  affine.store %c, %a[8]  : memref<16xi32>
  affine.store %c, %a[9]  : memref<16xi32>
  affine.store %c, %a[10] : memref<16xi32>
  affine.store %c, %a[11] : memref<16xi32>
  affine.store %c, %a[12] : memref<16xi32>
  affine.store %c, %a[13] : memref<16xi32>
  affine.store %c, %a[14] : memref<16xi32>
  affine.store %c, %a[15] : memref<16xi32>
  %v = affine.load %a[0]  : memref<16xi32>
  "use.consume"(%v) : (i32) -> ()
  return
}
