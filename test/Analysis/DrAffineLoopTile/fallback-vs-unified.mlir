// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-affine-loop-tile{use-unified-cost-model=false}))' | FileCheck %s

// Disable the unified path — fall back to upstream's nth-root placeholder.
// Verify the pass still emits a valid tiled nest.

// CHECK-LABEL: func.func @matmul
// CHECK-COUNT-6: affine.for
func.func @matmul(%A: memref<128x128xf32>, %B: memref<128x128xf32>, %C: memref<128x128xf32>) {
  affine.for %i = 0 to 128 {
    affine.for %j = 0 to 128 {
      affine.for %k = 0 to 128 {
        %a = affine.load %A[%i, %k] : memref<128x128xf32>
        %b = affine.load %B[%k, %j] : memref<128x128xf32>
        %c = affine.load %C[%i, %j] : memref<128x128xf32>
        %mul = arith.mulf %a, %b : f32
        %add = arith.addf %c, %mul : f32
        affine.store %add, %C[%i, %j] : memref<128x128xf32>
      }
    }
  }
  return
}
