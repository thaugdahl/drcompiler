// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-partial-remat dr-partial-max-leaves=2 dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// C036: Partial remat — leaf budget exceeded.
// dr-partial-max-leaves=2, but the expression tree has 3 partial-leaf loads.
// Partial remat cannot fire; the loads survive.

module {
  func.func @stride_leaf_budget() -> f32 {
    %c0 = arith.constant 0 : index

    // expected-remark @+1 {{cost-model:}}
    %l1 = memref.alloc() : memref<4xf32>
    // expected-remark @+1 {{cost-model:}}
    %l2 = memref.alloc() : memref<4xf32>
    // expected-remark @+1 {{cost-model:}}
    %l3 = memref.alloc() : memref<4xf32>
    // expected-remark @+1 {{cost-model:}}
    %dst = memref.alloc() : memref<1048576xf32>

    // Each leaf buffer written by loop-IV-dependent code.
    affine.for %i = 0 to 4 {
      %i32 = arith.index_cast %i : index to i32
      %f = arith.sitofp %i32 : i32 to f32
      affine.store %f, %l1[%i] : memref<4xf32>
      affine.store %f, %l2[%i] : memref<4xf32>
      affine.store %f, %l3[%i] : memref<4xf32>
    }

    // Producer: 3 leaf loads feed an add chain → exceeds max-leaves=2.
    affine.for %j = 0 to 1048576 {
      // expected-remark @below {{full-remat: REJECT_UNSAFE}}
      // expected-remark @below {{load: SINGLE}}
      %v1 = affine.load %l1[0] : memref<4xf32>
      // expected-remark @below {{full-remat: REJECT_UNSAFE}}
      // expected-remark @below {{load: SINGLE}}
      %v2 = affine.load %l2[0] : memref<4xf32>
      // expected-remark @below {{full-remat: REJECT_UNSAFE}}
      // expected-remark @below {{load: SINGLE}}
      %v3 = affine.load %l3[0] : memref<4xf32>
      %s1 = arith.addf %v1, %v2 : f32
      %s2 = arith.addf %s1, %v3 : f32
      affine.store %s2, %dst[%j] : memref<1048576xf32>
    }

    // expected-remark @below {{full-remat: REJECT_UNSAFE}}
    // expected-remark @below {{load: SINGLE}}
    %out = affine.load %dst[%c0] : memref<1048576xf32>
    memref.dealloc %l1 : memref<4xf32>
    memref.dealloc %l2 : memref<4xf32>
    memref.dealloc %l3 : memref<4xf32>
    memref.dealloc %dst : memref<1048576xf32>
    return %out : f32
  }
}

// Loads survive — leaf budget exceeded with max-leaves=2.
// CHECK-LABEL: func.func @stride_leaf_budget
// CHECK:         affine.for
// CHECK:           affine.load %{{.*}}[0] : memref<4xf32>
// CHECK:           affine.load %{{.*}}[0] : memref<4xf32>
// CHECK:           affine.load %{{.*}}[0] : memref<4xf32>
