// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block{mr=2 nr=2 cache-tile=true mc=16 nc=16 kc=16}))' | FileCheck %s

// With cache-tile, the GEMM band is first blocked by mc x nc x kc (here 16^3),
// producing cache-tile loops stepped by the tile size; the register-block
// micro-kernel then runs on the cache-resident point loops (iter_args).

module {
  func.func @gemm(%A: memref<64x64xf64>, %B: memref<64x64xf64>, %C: memref<64x64xf64>) {
    affine.for %i = 0 to 64 {
      affine.for %j = 0 to 64 {
        affine.for %k = 0 to 64 {
          %a = affine.load %A[%i, %k] : memref<64x64xf64>
          %b = affine.load %B[%k, %j] : memref<64x64xf64>
          %c = affine.load %C[%i, %j] : memref<64x64xf64>
          %p = arith.mulf %a, %b : f64
          %s = arith.addf %c, %p : f64
          affine.store %s, %C[%i, %j] : memref<64x64xf64>
        }
      }
    }
    return
  }
}

// Cache-tile loops stepped by the tile size (16).
// CHECK: affine.for %{{.*}} = 0 to 64 step 16
// CHECK:   affine.for %{{.*}} = 0 to 64 step 16
// CHECK:     affine.for %{{.*}} = 0 to 64 step 16

// The register-block micro-kernel: a reduction loop carrying 2x2 iter_args.
// CHECK: affine.for %{{.*}} iter_args({{.*}}) -> (f64, f64, f64, f64)
// CHECK:   affine.yield %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f64, f64, f64, f64
