// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block{mr=2 nr=2 cache-tile=true mc=16 nc=16 kc=16}))' | FileCheck %s

// With cache-tile, the GEMM band is first blocked by mc x nc x kc (here 16^3),
// producing cache-tile loops stepped by the tile size; the register-block
// micro-kernel then runs on the cache-resident point loops.  The micro-kernel is
// EXPLICITLY vectorized in the vector dialect (no reliance on LLVM-SLP) even for
// this 2D gemm: mr=2 vector<8xf64> accumulators carried over the reduction, with
// vector.broadcast of the streamed operand and FMA-contractable arith.

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

// Explicit-vector register-block micro-kernel on the cache-resident point loops:
// inner spatial loop stepped by vl=8, two vector<8xf64> accumulators carried over
// the reduction, vector.broadcast + FMA-contractable vector arith, vector store.
// CHECK:       affine.for %{{.*}} step 8
// CHECK:         affine.vector_load %{{.*}} : memref<64x64xf64>, vector<8xf64>
// CHECK:         affine.for %{{.*}} iter_args({{.*}}) -> (vector<8xf64>, vector<8xf64>)
// CHECK:           vector.broadcast %{{.*}} : f64 to vector<8xf64>
// CHECK:           arith.mulf %{{.*}}, %{{.*}} fastmath<contract> : vector<8xf64>
// CHECK:           arith.addf %{{.*}}, %{{.*}} fastmath<contract> : vector<8xf64>
// CHECK:           affine.yield %{{.*}}, %{{.*}} : vector<8xf64>, vector<8xf64>
// CHECK:         affine.vector_store %{{.*}}, %{{.*}} : memref<64x64xf64>, vector<8xf64>
