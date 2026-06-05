// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block{mr=2 nr=2 vectorize=false}))' | FileCheck %s

// A perfectly-nested affine GEMM (the canonical cgeist-emitted shape, with the
// C[i,j] accumulator carried in memory across the k reduction) is register
// blocked: the two outer loops are stepped by mr x nr, the mr*nr accumulators
// are loaded before the reduction loop, carried as iter_args, and stored after.
//
// vectorize=false pins the SCALAR register-block (mr*nr scalar iter_args + LLVM
// SLP) path.  The default vector-dialect micro-kernel is covered by
// gemm-vector-kernel.mlir.

module {
  func.func @gemm(%A: memref<8x8xf64>, %B: memref<8x8xf64>, %C: memref<8x8xf64>) {
    affine.for %i = 0 to 8 {
      affine.for %j = 0 to 8 {
        affine.for %k = 0 to 8 {
          %a = affine.load %A[%i, %k] : memref<8x8xf64>
          %b = affine.load %B[%k, %j] : memref<8x8xf64>
          %c = affine.load %C[%i, %j] : memref<8x8xf64>
          %p = arith.mulf %a, %b : f64
          %s = arith.addf %c, %p : f64
          affine.store %s, %C[%i, %j] : memref<8x8xf64>
        }
      }
    }
    return
  }
}

// The outer loops step by the register-block tile.
// CHECK: affine.for %{{.*}} = 0 to 8 step 2
// CHECK:   affine.for %{{.*}} = 0 to 8 step 2

// The 2x2 = 4 accumulators are loaded from C before the reduction loop.
// CHECK:     affine.load %{{.*}}[%{{.*}}, %{{.*}}]
// CHECK:     affine.load %{{.*}}[%{{.*}}, %{{.*}}]
// CHECK:     affine.load %{{.*}}[%{{.*}}, %{{.*}}]
// CHECK:     affine.load %{{.*}}[%{{.*}}, %{{.*}}]

// The reduction loop carries the 4 accumulators as iter_args and yields them;
// there is NO affine.store inside the reduction loop body.
// CHECK:     affine.for %{{.*}} = 0 to 8 iter_args({{.*}}) -> (f64, f64, f64, f64)
// CHECK-NOT:   affine.store
// CHECK:       affine.yield %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f64, f64, f64, f64

// The 4 final accumulators are stored back to C after the reduction loop.
// CHECK:     affine.store %{{.*}}#{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}]
// CHECK:     affine.store %{{.*}}#{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}]
// CHECK:     affine.store %{{.*}}#{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}]
// CHECK:     affine.store %{{.*}}#{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}]
