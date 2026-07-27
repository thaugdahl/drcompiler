// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-partial-remat dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// B022: Partial remat — leaf from memref.global constant.
// The constant global is recognized as a safe leaf for partial remat.
// With partial-remat enabled, the %out load is rematerialized via
// partial-remat rather than constant-global-fold.

module {
  memref.global "private" constant @weights : memref<4xf32> = dense<[1.0, 2.0, 3.0, 4.0]>

  func.func @partial_global_const_leaf() -> f32 {
    %c0 = arith.constant 0 : index
    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %dst = memref.alloc() : memref<1048576xf32>

    %w = memref.get_global @weights : memref<4xf32>

    affine.for %j = 0 to 1048576 {
      %g = affine.load %w[0] : memref<4xf32>
      affine.store %g, %dst[%j] : memref<1048576xf32>
    }

    // expected-remark @below {{load: SINGLE}}
    // expected-remark @below {{full-remat: REJECT_UNSAFE}}
    // expected-remark @below {{partial-remat: ACCEPT}}
    %out = affine.load %dst[%c0] : memref<1048576xf32>
    memref.dealloc %dst : memref<1048576xf32>
    return %out : f32
  }
}

// After partial remat: the %out load is replaced by a cloned leaf load from the const global.
// CHECK-LABEL: func.func @partial_global_const_leaf
// CHECK:         affine.for
// CHECK:         }
// CHECK:         %[[LEAF:.*]] = affine.load %{{.*}}[0] : memref<4xf32>
// CHECK:         return %[[LEAF]] : f32
