// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block{cache-tile=true mc=64 nc=64 kc=64}))' | FileCheck %s

// Family auto-selection (family-select defaults true).  The rank-k @syrk nest --
// multiplicands A[i][k] and A[j][k] are both stride-1 in the reduction k, none
// stride-1 in the spatial j -- is detected as the DOT family.  The pass then
// overrides the tile to a small square (mr=nr=4 here -> strip step 4) so the
// accumulator grid fits the vector register file, and sets fastmath<fast> on the
// reduction FP ops so the LLVM backend can reassociate and vectorize the
// k-reduction (without which it stays scalar -- the rank-k "loss" was exactly
// this).  Contrast: broadcast family keeps the wide tile and no reassociation
// (gemm-no-reassoc.mlir).

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

// Dot family: small square tile (strip step 4) + fastmath<fast> on the
// register-blocked reduction ops.
// CHECK-LABEL: func.func @syrk
// CHECK: affine.for %{{.*}} = 0 to 8 step 4
// CHECK: arith.mulf %{{.*}}, %{{.*}} fastmath<fast>
// CHECK: arith.addf %{{.*}}, %{{.*}} fastmath<fast>
