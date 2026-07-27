// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-partial-remat dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// B017: Partial remat — chain with 1 non-chainable leaf load.
// %src is written by a loop-IV-dependent store (breaks chain).
// The downstream %out load triggers partial remat with 1 leaf.

module {
  func.func @partial_one_leaf() -> f32 {
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

// CHECK-LABEL: func.func @partial_one_leaf
// CHECK:         affine.for
// CHECK:         affine.for
// CHECK:         }
// CHECK-NEXT:    %[[LEAF:.*]] = affine.load %{{.*}}[0] : memref<4xf32>
// CHECK:         %[[SUM:.*]] = arith.addf %[[LEAF]]
// CHECK:         return %[[SUM]] : f32
