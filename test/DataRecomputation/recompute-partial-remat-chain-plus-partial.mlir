// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-partial-remat dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// Tree mixes a trivially-chainable leaf (%chain, constant store) with a
// partial-leaf candidate (%partial, loop-IV-dependent store that breaks
// the chain path).  Partial-remat ACCEPTs; the chain leaf collapses to
// its constant store value (no clone of memref.load), while the partial
// leaf is cloned as an affine.load.  Phase 1 (S1-A) tightened analysis
// precision so the chain propagates the constant inline rather than
// emitting a cloned load.

module {
  func.func @test() -> f32 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f32
    // expected-remark @+1 {{cost-model:}}
    %chain = memref.alloc() : memref<f32>
    // expected-remark @+1 {{cost-model:}}
    %partial = memref.alloc() : memref<4xf32>
    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %dst = memref.alloc() : memref<1048576xf32>

    // Simple store to %chain.
    memref.store %one, %chain[] : memref<f32>

    // Loop-IV-dependent store to %partial — breaks chain path.
    affine.for %i = 0 to 4 {
      %i32 = arith.index_cast %i : index to i32
      %f = arith.sitofp %i32 : i32 to f32
      affine.store %f, %partial[%i] : memref<4xf32>
    }

    affine.for %j = 0 to 1048576 {
      // expected-remark @+1 {{load: SINGLE}}
      %vc = memref.load %chain[] : memref<f32>
      // expected-remark @+1 {{load: SINGLE}}
      %vp = affine.load %partial[0] : memref<4xf32>
      %s = arith.addf %vc, %vp : f32
      affine.store %s, %dst[%j] : memref<1048576xf32>
    }

    // expected-remark @+2 {{load: SINGLE}}
    // expected-remark @+1 {{partial-remat: ACCEPT}}
    %out = affine.load %dst[%c0] : memref<1048576xf32>
    memref.dealloc %chain : memref<f32>
    memref.dealloc %partial : memref<4xf32>
    memref.dealloc %dst : memref<1048576xf32>
    return %out : f32
  }
}

// Rematerialized expression: chain leaf folded to constant, partial leaf
// cloned as affine.load, addf feeding the return.
// CHECK-LABEL: func.func @test
// CHECK:         affine.store %{{.*}}, %{{.*}} : memref<1048576xf32>
// CHECK:         %[[CST:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:    %[[LP:.*]] = affine.load %{{.*}}[0] : memref<4xf32>
// CHECK-NEXT:    %[[SUM:.*]] = arith.addf %[[CST]], %[[LP]] : f32
// CHECK:         return %[[SUM]] : f32
