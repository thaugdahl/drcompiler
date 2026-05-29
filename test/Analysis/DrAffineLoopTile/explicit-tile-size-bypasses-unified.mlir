// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-affine-loop-tile{tile-size=8}))' | FileCheck %s

// Explicit --tile-size short-circuits the grid search.  Verify the tile
// step size of 8 appears in the outer-band loop headers.

// CHECK-LABEL: func.func @matmul
// CHECK: affine.for {{.*}} = 0 to 64 step 8
func.func @matmul(%A: memref<64x64xf32>, %B: memref<64x64xf32>, %C: memref<64x64xf32>) {
  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 64 {
      affine.for %k = 0 to 64 {
        %a = affine.load %A[%i, %k] : memref<64x64xf32>
        %b = affine.load %B[%k, %j] : memref<64x64xf32>
        %c = affine.load %C[%i, %j] : memref<64x64xf32>
        %mul = arith.mulf %a, %b : f32
        %add = arith.addf %c, %mul : f32
        affine.store %add, %C[%i, %j] : memref<64x64xf32>
      }
    }
  }
  return
}
