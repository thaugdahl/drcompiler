// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block{mr=2 nr=2 cache-tile=true mc=64 nc=64 kc=64 family-select=false}))' | FileCheck %s
//
// family-select=false pins the raw transform at the explicit 2x2 tile (this
// test checks the structural diagonal-peel + register-block).  Auto family
// selection is covered by syrk-family-select.mlir.

// A triangular rank-k reduction nest (syrk shape: inner spatial bound j:0..i+1
// depends on the outer IV i, and the second operand A[j][k] is a transpose
// access).  The pass DIAGONAL-PEELS it: strip-mine i by mr, split j into a
// rectangular HEAD (j in [0,ii), register-blocked) plus a scalar DIAGONAL
// remainder (j in [ii, i+1)).  This proves the triangular nest register-blocks
// and stays correct -- though the spike (SPIKE_GENERALIZATION_FINDINGS.md) shows
// it is NOT profitable vs clang's k-vectorized dot product for rank-k kernels.

module {
  func.func @syrk(%A: memref<8x8xf64>, %C: memref<8x8xf64>) {
    affine.for %i = 0 to 8 {
      affine.for %j = 0 to affine_map<(d0) -> (d0 + 1)>(%i) {
        affine.for %k = 0 to 8 {
          %a = affine.load %A[%i, %k] : memref<8x8xf64>
          %b = affine.load %A[%j, %k] : memref<8x8xf64>
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

// Strip-mined i (step mr=2) with the register-blocked rectangular HEAD:
// CHECK: affine.for %{{.*}} = 0 to 8 step 2
// CHECK:   affine.for %{{.*}} = 0 to #{{.*}}(%{{.*}}) step 2
// CHECK:     affine.for %{{.*}} = 0 to 8 iter_args({{.*}}) -> (f64, f64, f64, f64)
// CHECK:       affine.yield %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f64, f64, f64, f64

// Scalar DIAGONAL remainder (ragged j upper bound, not register-blocked):
// CHECK: affine.for %{{.*}} = #{{.*}}(%{{.*}}) to #{{.*}}(%{{.*}}) {
