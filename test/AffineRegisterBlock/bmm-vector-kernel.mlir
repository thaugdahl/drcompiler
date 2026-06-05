// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block{mr=2 vl=8}))' | FileCheck %s

// Batched matmul C[b][i][j] = sum_k A[b][i][k]*B[b][k][j] -- a broadcast
// contraction with a 3D (>2D) accumulator.  LLVM SLP fails to vectorize the
// micro-kernel through the extra batch index, so the pass emits an EXPLICIT
// vector-dialect micro-kernel (reserved for >2D accumulators; 2D-output stays on
// the scalar+SLP path).  The outer spatial loop i is unroll-jammed by mr; the
// inner spatial loop j is vectorized to vector<vl>: mr vector accumulators
// carried over k, the streamed operand a contiguous affine.vector_load, the
// broadcast operand a scalar load + vector.broadcast, each FMA a vector op.

module {
  func.func @bmm(%A: memref<2x8x8xf64>, %B: memref<2x8x8xf64>, %C: memref<2x8x8xf64>) {
    affine.for %b = 0 to 2 {
      affine.for %i = 0 to 8 {
        affine.for %j = 0 to 8 {
          affine.for %k = 0 to 8 {
            %a = affine.load %A[%b, %i, %k] : memref<2x8x8xf64>
            %bb = affine.load %B[%b, %k, %j] : memref<2x8x8xf64>
            %c = affine.load %C[%b, %i, %j] : memref<2x8x8xf64>
            %p = arith.mulf %a, %bb : f64
            %s = arith.addf %c, %p : f64
            affine.store %s, %C[%b, %i, %j] : memref<2x8x8xf64>
          }
        }
      }
    }
    return
  }
}

// Inner spatial j vectorized (vl=8), outer spatial i unroll-jammed by mr=2.
// CHECK: affine.for %{{.*}} = 0 to 8 step 2
// CHECK: affine.vector_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<2x8x8xf64>, vector<8xf64>
// CHECK: affine.for %{{.*}} = 0 to 8 iter_args({{.*}}) -> (vector<8xf64>, vector<8xf64>)
// CHECK: vector.broadcast %{{.*}} : f64 to vector<8xf64>
// CHECK: arith.mulf %{{.*}}, %{{.*}} : vector<8xf64>
// CHECK: affine.yield %{{.*}}, %{{.*}} : vector<8xf64>, vector<8xf64>
// CHECK: affine.vector_store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<2x8x8xf64>, vector<8xf64>
