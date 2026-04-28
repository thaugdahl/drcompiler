// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-partial-remat dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// Partial rematerialization accepts a leaf load of a write-once buffer.
//
// Chain strategy fails here: %src's stored value %f depends on the
// src-loop IV %i, which does not dominate the downstream %out load.
// Strategy 2b treats the (loop-invariant) load of %src[0] as a partial
// leaf and clones it verbatim.  The load of %dst is large (L3), so the
// cost gate accepts the swap (1 leaf load + 1 addf < L3 load latency).

module {
  func.func @test() -> f32 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f32
    // expected-remark @+1 {{cost-model:}}
    %src = memref.alloc() : memref<4xf32>
    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %dst = memref.alloc() : memref<1048576xf32>

    affine.for %i = 0 to 4 {
      %i32 = arith.index_cast %i : index to i32
      %f = arith.sitofp %i32 : i32 to f32
      affine.store %f, %src[%i] : memref<4xf32>
    }

    affine.for %j = 0 to 1048576 {
      // expected-remark @below {{full-remat: REJECT_UNSAFE}}
      // expected-remark @below {{load: SINGLE}}
      %v = affine.load %src[0] : memref<4xf32>
      %s = arith.addf %v, %one : f32
      affine.store %s, %dst[%j] : memref<1048576xf32>
    }

    // expected-remark @below {{load: SINGLE}}
    // expected-remark @below {{full-remat: REJECT_UNSAFE}}
    // expected-remark @below {{partial-remat: ACCEPT}}
    %out = affine.load %dst[%c0] : memref<1048576xf32>
    memref.dealloc %src : memref<4xf32>
    memref.dealloc %dst : memref<1048576xf32>
    return %out : f32
  }
}

// After partial remat, the downstream %out load is gone; remat contains a
// cloned load of %src[0] plus the addf. The final return takes the addf
// result, not a reload of the large buffer.
// CHECK-LABEL: func.func @test
// CHECK:         affine.for
// CHECK:         affine.for
// CHECK:         }
// CHECK-NEXT:    %[[LEAF:.*]] = affine.load %{{.*}}[0] : memref<4xf32>
// CHECK:         %[[SUM:.*]] = arith.addf %[[LEAF]]
// CHECK:         return %[[SUM]] : f32
