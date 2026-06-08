// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block{mr=4 nr=4}))' | FileCheck %s

// Dot (rank-k) family with an EXPLICIT reduction-vector micro-kernel and a
// k-tail peel.  Multiplicands A[i][k], A[j][k] are stride-1 in the reduction k,
// so the pass vectorizes the reduction itself: each accumulator becomes a
// vector<8xf64> partial sum carried over k, horizontal-reduced (vector.reduction)
// at the end -- NOT a bet on LLVM reduction-vectorizing a scalar loop.  Here the
// reduction trip is 20, not a multiple of vl=8, so the k-loop is peeled into a
// vl-divisible vector MAIN ([0,16) step 8) plus a scalar TAIL ([16,20)) that
// accumulates the leftover k into the same C after the main stores.

module {
  func.func @syrk(%A: memref<8x20xf64>, %C: memref<8x8xf64>) {
    affine.for %i = 0 to 8 {
      affine.for %j = 0 to 8 {
        affine.for %k = 0 to 20 {
          %a = affine.load %A[%i, %k] : memref<8x20xf64>
          %b = affine.load %A[%j, %k] : memref<8x20xf64>
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

// CHECK-LABEL: func.func @syrk
// Vector MAIN over the vl-divisible part of k, carrying vector<8xf64> partial sums.
// CHECK: affine.for %{{.*}} = 0 to 16 step 8 iter_args
// CHECK:   arith.mulf %{{.*}}, %{{.*}} fastmath<fast> : vector<8xf64>
// CHECK:   arith.addf %{{.*}}, %{{.*}} fastmath<fast> : vector<8xf64>
// Horizontal reduce each partial sum back to a scalar, add C, store.
// CHECK: vector.reduction <add>, %{{.*}} fastmath<fast> : vector<8xf64> into f64
// Scalar TAIL over the leftover k, accumulating into the same C.
// CHECK: affine.for %{{.*}} = 16 to 20
// CHECK:   arith.mulf %{{.*}}, %{{.*}} : f64
