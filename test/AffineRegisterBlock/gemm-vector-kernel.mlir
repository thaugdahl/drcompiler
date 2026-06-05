// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block{mr=2 vl=8}))' | FileCheck %s

// The broadcast family (C=A*B) is register-blocked with an EXPLICIT vector
// micro-kernel (the default; family auto-selection picks it).  Instead of
// unroll-jamming the inner spatial loop by nr and relying on LLVM SLP -- which
// fails on the >2D accumulator addressing of tensor contractions (batched
// matmul) -- the inner spatial loop is vectorized to vector<vl>: the outer
// spatial loop is unroll-jammed by mr (mr vector accumulators), the streamed
// operand is a contiguous affine.vector_load, the broadcast operand a scalar
// load + vector.broadcast, and each FMA a vector op.  This makes vectorization
// explicit so it survives arbitrary tensor-contraction addressing.

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

// Outer spatial loop unroll-jammed by mr=2; inner spatial vectorized (vl=8).
// CHECK: affine.for %{{.*}} = 0 to 8 step 2
// The mr accumulators are loaded VL-wide before the reduction.
// CHECK: affine.vector_load %{{.*}}[%{{.*}}, %{{.*}}] : memref<8x8xf64>, vector<8xf64>
// The reduction carries mr vector accumulators (no scalar iter_args, no store
// inside the loop).
// CHECK: affine.for %{{.*}} = 0 to 8 iter_args({{.*}}) -> (vector<8xf64>, vector<8xf64>)
// The streamed operand is a contiguous vector load; the broadcast operand a
// scalar load broadcast to a vector; the FMA is a vector multiply/add.
// CHECK: affine.vector_load %{{.*}} : memref<8x8xf64>, vector<8xf64>
// CHECK: vector.broadcast %{{.*}} : f64 to vector<8xf64>
// CHECK: arith.mulf %{{.*}}, %{{.*}} : vector<8xf64>
// CHECK: affine.yield %{{.*}}, %{{.*}} : vector<8xf64>, vector<8xf64>
// The vector results are stored back after the reduction.
// CHECK: affine.vector_store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<8x8xf64>, vector<8xf64>
