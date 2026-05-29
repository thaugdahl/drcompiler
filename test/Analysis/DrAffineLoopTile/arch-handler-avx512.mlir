// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-affine-loop-tile{arch-handler=x86-64-avx512 spill-strategy=graph-color}))' | FileCheck %s

// arch-handler + spill-strategy plumb through.  Pass should not crash and
// should still produce a tiled nest.

// CHECK-LABEL: func.func @stencil
// CHECK-COUNT-4: affine.for
func.func @stencil(%A: memref<512x512xf32>, %B: memref<512x512xf32>) {
  affine.for %i = 1 to 511 {
    affine.for %j = 1 to 511 {
      %l = affine.load %A[%i - 1, %j] : memref<512x512xf32>
      %r = affine.load %A[%i + 1, %j] : memref<512x512xf32>
      %s = arith.addf %l, %r : f32
      affine.store %s, %B[%i, %j] : memref<512x512xf32>
    }
  }
  return
}
