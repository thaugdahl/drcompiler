// RUN: dr-opt %s --pass-pipeline='builtin.module(dr-affine-loop-fusion{use-unified-cost-model=false})' | FileCheck %s

// Same case but with the unified cost model explicitly disabled — the pass
// falls back to upstream's storage-reduction-based placeholder.  Verify the
// pass still functions and fuses cleanly.

// CHECK-LABEL: func.func @producer_consumer_fallback
// CHECK:         affine.for %{{.*}} = 0 to 16
// CHECK-NOT:     affine.for
// CHECK:         return

func.func @producer_consumer_fallback(%A: memref<16xf32>, %B: memref<16xf32>) {
  %t = memref.alloc() : memref<16xf32>
  %c = arith.constant 1.0 : f32
  affine.for %i = 0 to 16 {
    %v = affine.load %A[%i] : memref<16xf32>
    %r = arith.mulf %v, %c : f32
    affine.store %r, %t[%i] : memref<16xf32>
  }
  affine.for %i = 0 to 16 {
    %v = affine.load %t[%i] : memref<16xf32>
    %r = arith.addf %v, %c : f32
    affine.store %r, %B[%i] : memref<16xf32>
  }
  memref.dealloc %t : memref<16xf32>
  return
}
