// RUN: dr-opt %s --dr-affine-loop-fusion | FileCheck %s

// Classical producer-consumer fusion case.  Under the default
// (use-unified-cost-model=true) path our fork should still fuse the two
// loops — the unified combiner agrees with upstream when register pressure
// is trivially under budget.

// CHECK-LABEL: func.func @producer_consumer
// CHECK:         affine.for %{{.*}} = 0 to 10
// CHECK-NOT:     affine.for
// CHECK:         return

func.func @producer_consumer(%A: memref<10xf32>, %B: memref<10xf32>) {
  %t = memref.alloc() : memref<10xf32>
  %c = arith.constant 1.0 : f32
  affine.for %i = 0 to 10 {
    %v = affine.load %A[%i] : memref<10xf32>
    %r = arith.mulf %v, %c : f32
    affine.store %r, %t[%i] : memref<10xf32>
  }
  affine.for %i = 0 to 10 {
    %v = affine.load %t[%i] : memref<10xf32>
    %r = arith.addf %v, %c : f32
    affine.store %r, %B[%i] : memref<10xf32>
  }
  memref.dealloc %t : memref<10xf32>
  return
}
