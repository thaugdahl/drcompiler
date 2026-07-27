// RUN: dr-opt %s --allow-unregistered-dialect --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-cost-model=true dr-buffer-elim=true dr-summary=true})' 2>&1 | FileCheck %s --check-prefix=SUM

// E019: 4 stores to different indices, all writing the same computation
// (arith.addi). 4 matching loads. The structural hash detects all stores
// share the same tree → distinctTrees = 1. elimCost uses
// numDistinctComputes = min(4, 1) = 1, so only 1 × computeCost is charged
// instead of 4 × computeCost. Shared subexpr discount makes elimination
// cheaper.

// SUM: buffer-elim {{.*}}: FEASIBLE (escape=no-escape, loads=4
func.func @shared_discount(%x: i32) {
  %a = memref.alloca() : memref<4xi32>
  %sum = arith.addi %x, %x : i32
  affine.store %sum, %a[0] : memref<4xi32>
  affine.store %sum, %a[1] : memref<4xi32>
  affine.store %sum, %a[2] : memref<4xi32>
  affine.store %sum, %a[3] : memref<4xi32>
  %v0 = affine.load %a[0] : memref<4xi32>
  %v1 = affine.load %a[1] : memref<4xi32>
  %v2 = affine.load %a[2] : memref<4xi32>
  %v3 = affine.load %a[3] : memref<4xi32>
  "use.consume"(%v0, %v1, %v2, %v3) : (i32, i32, i32, i32) -> ()
  return
}
