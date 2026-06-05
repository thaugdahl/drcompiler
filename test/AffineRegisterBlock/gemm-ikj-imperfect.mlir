// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block{mr=2 nr=2}))' | FileCheck %s

// PolyBench-shaped GEMM: the i-loop imperfectly nests a beta-scaling sibling
// loop and an i-k-j matmul whose reduction (k) is the MIDDLE loop, not the
// innermost.  The pass must (1) leave the beta loop intact, (2) interchange
// k<->j so the reduction is innermost, (3) register-block the matmul.

module {
  func.func @gemm(%A: memref<4x4xf64>, %B: memref<4x4xf64>, %C: memref<4x4xf64>,
                  %beta: f64) {
    affine.for %i = 0 to 4 {
      // beta scaling: C[i,j] *= beta   (sibling loop -- must survive)
      affine.for %j = 0 to 4 {
        %c = affine.load %C[%i, %j] : memref<4x4xf64>
        %s = arith.mulf %c, %beta : f64
        affine.store %s, %C[%i, %j] : memref<4x4xf64>
      }
      // matmul in i-k-j order: reduction over k is the middle loop.
      affine.for %k = 0 to 4 {
        affine.for %j = 0 to 4 {
          %a = affine.load %A[%i, %k] : memref<4x4xf64>
          %b = affine.load %B[%k, %j] : memref<4x4xf64>
          %c = affine.load %C[%i, %j] : memref<4x4xf64>
          %p = arith.mulf %a, %b : f64
          %s = arith.addf %c, %p : f64
          affine.store %s, %C[%i, %j] : memref<4x4xf64>
        }
      }
    }
    return
  }
}

// The beta-scaling sibling loop is preserved.
// CHECK: affine.for %{{.*}} = 0 to 4 step 2
// CHECK:   affine.for
// CHECK:     arith.mulf %{{.*}}, %{{.*}} : f64

// The matmul is register-blocked: a reduction loop carrying 2x2 = 4 iter_args,
// with no store inside it, and the accumulators sunk afterwards.
// CHECK: affine.for %{{.*}} = 0 to 4 iter_args({{.*}}) -> (f64, f64, f64, f64)
// CHECK-NOT:   affine.store
// CHECK:       affine.yield %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f64, f64, f64, f64
// CHECK:     affine.store
