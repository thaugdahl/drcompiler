// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block{mr=2 nr=2 cache-tile=true mc=16 nc=16 kc=16}))' | FileCheck %s

// PolyBench-shaped GEMM (imperfect i-nest: a beta-scaling sibling loop + an
// i-k-j matmul).  With cache-tile, the pass must first DISTRIBUTE the i-loop so
// the matmul becomes a perfect band, then cache-tile + register-block it -- while
// leaving the beta loop as its own sibling i-loop running first.

module {
  func.func @gemm(%A: memref<32x32xf64>, %B: memref<32x32xf64>, %C: memref<32x32xf64>,
                  %beta: f64) {
    affine.for %i = 0 to 32 {
      affine.for %j = 0 to 32 {
        %c = affine.load %C[%i, %j] : memref<32x32xf64>
        %s = arith.mulf %c, %beta : f64
        affine.store %s, %C[%i, %j] : memref<32x32xf64>
      }
      affine.for %k = 0 to 32 {
        affine.for %j = 0 to 32 {
          %a = affine.load %A[%i, %k] : memref<32x32xf64>
          %b = affine.load %B[%k, %j] : memref<32x32xf64>
          %c = affine.load %C[%i, %j] : memref<32x32xf64>
          %p = arith.mulf %a, %b : f64
          %s = arith.addf %c, %p : f64
          affine.store %s, %C[%i, %j] : memref<32x32xf64>
        }
      }
    }
    return
  }
}

// First sibling i-loop: the beta scaling, distributed out and left untiled.
// CHECK: affine.for %{{.*}} = 0 to 32 {
// CHECK:   affine.for %{{.*}} = 0 to 32 {
// CHECK:     arith.mulf %{{.*}}, %{{.*}} : f64
// CHECK:     affine.store

// Second sibling i-loop: the matmul, cache-tiled (step 16) and register-blocked.
// CHECK: affine.for %{{.*}} = 0 to 32 step 16
// CHECK:   affine.for %{{.*}} = 0 to 32 step 16
// CHECK:     affine.for %{{.*}} = 0 to 32 step 16
// CHECK: affine.for %{{.*}} iter_args({{.*}}) -> (f64, f64, f64, f64)
// CHECK:   affine.yield %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f64, f64, f64, f64
