// RUN: dr-opt %s --pass-pipeline='builtin.module(dr-affine-loop-fusion{spill-strategy=graph-color})' | FileCheck %s

// Smoke-test the graph-color spill strategy through the fusion pass.  The
// strategy choice does not flip this trivial case but the pipeline must
// not crash and the fusion still fires.

// CHECK-LABEL: func.func @gc_smoke
// CHECK:         affine.for %{{.*}} = 0 to 8
// CHECK-NOT:     affine.for
// CHECK:         return

func.func @gc_smoke(%A: memref<8xf32>, %B: memref<8xf32>) {
  %t = memref.alloc() : memref<8xf32>
  %c = arith.constant 0.5 : f32
  affine.for %i = 0 to 8 {
    %v = affine.load %A[%i] : memref<8xf32>
    %r = arith.mulf %v, %c : f32
    affine.store %r, %t[%i] : memref<8xf32>
  }
  affine.for %i = 0 to 8 {
    %v = affine.load %t[%i] : memref<8xf32>
    %r = arith.addf %v, %c : f32
    affine.store %r, %B[%i] : memref<8xf32>
  }
  memref.dealloc %t : memref<8xf32>
  return
}
