// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-partial-remat dr-partial-max-leaves=4 dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// Five partial-leaf candidates in the tree exceed the configured
// dr-partial-max-leaves=4 budget; partial remat must bail and leave
// the IR unchanged.

module {
  func.func @test() -> f32 {
    %c0 = arith.constant 0 : index
    // expected-remark @+1 {{cost-model:}}
    %l1 = memref.alloc() : memref<4xf32>
    // expected-remark @+1 {{cost-model:}}
    %l2 = memref.alloc() : memref<4xf32>
    // expected-remark @+1 {{cost-model:}}
    %l3 = memref.alloc() : memref<4xf32>
    // expected-remark @+1 {{cost-model:}}
    %l4 = memref.alloc() : memref<4xf32>
    // expected-remark @+1 {{cost-model:}}
    %l5 = memref.alloc() : memref<4xf32>
    // expected-remark @+1 {{cost-model:}}
    %dst = memref.alloc() : memref<1048576xf32>

    // Each %lN is written by a loop whose store value depends on the
    // loop IV, breaking chain rematerialization.
    affine.for %i = 0 to 4 {
      %i32 = arith.index_cast %i : index to i32
      %f = arith.sitofp %i32 : i32 to f32
      affine.store %f, %l1[%i] : memref<4xf32>
      affine.store %f, %l2[%i] : memref<4xf32>
      affine.store %f, %l3[%i] : memref<4xf32>
      affine.store %f, %l4[%i] : memref<4xf32>
      affine.store %f, %l5[%i] : memref<4xf32>
    }

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
      // expected-remark @below {{full-remat: REJECT_UNSAFE}}
      // expected-remark @below {{load: SINGLE}}
      %v4 = affine.load %l4[0] : memref<4xf32>
      // expected-remark @below {{full-remat: REJECT_UNSAFE}}
      // expected-remark @below {{load: SINGLE}}
      %v5 = affine.load %l5[0] : memref<4xf32>
      %s1 = arith.addf %v1, %v2 : f32
      %s2 = arith.addf %s1, %v3 : f32
      %s3 = arith.addf %s2, %v4 : f32
      %s4 = arith.addf %s3, %v5 : f32
      affine.store %s4, %dst[%j] : memref<1048576xf32>
    }

    // expected-remark @below {{full-remat: REJECT_UNSAFE}}
    // expected-remark @below {{load: SINGLE}}
    %out = affine.load %dst[%c0] : memref<1048576xf32>
    memref.dealloc %l1 : memref<4xf32>
    memref.dealloc %l2 : memref<4xf32>
    memref.dealloc %l3 : memref<4xf32>
    memref.dealloc %l4 : memref<4xf32>
    memref.dealloc %l5 : memref<4xf32>
    memref.dealloc %dst : memref<1048576xf32>
    return %out : f32
  }
}

// IR unchanged — leaf budget exceeded.
// CHECK-LABEL: func.func @test
// CHECK:         affine.for
// CHECK:         affine.for
// CHECK:         %[[OUT:.*]] = affine.load %{{.*}}[%c0] : memref<1048576xf32>
// CHECK:         return %[[OUT]] : f32
