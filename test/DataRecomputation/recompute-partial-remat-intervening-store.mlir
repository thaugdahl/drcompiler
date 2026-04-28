// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-partial-remat dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// Partial-remat safety check rejects leaf loads whose allocation root
// has more than one textual writer; re-reading would race with any
// intervening store.

module {
  func.func @test() -> f32 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f32
    %zero = arith.constant 0.0 : f32
    %src = memref.alloc() : memref<4xf32>
    // expected-remark @+1 {{cost-model:}}
    %dst = memref.alloc() : memref<1048576xf32>

    // First writer.
    affine.for %i = 0 to 4 {
      affine.store %zero, %src[%i] : memref<4xf32>
    }

    // Second writer — makes load of %src MULTI and rootWriteCount > 1.
    affine.for %i = 0 to 4 {
      %i32 = arith.index_cast %i : index to i32
      %f = arith.sitofp %i32 : i32 to f32
      affine.store %f, %src[%i] : memref<4xf32>
    }

    affine.for %j = 0 to 1048576 {
      // expected-remark @+1 {{load: MULTI}}
      %v = affine.load %src[0] : memref<4xf32>
      %s = arith.addf %v, %one : f32
      affine.store %s, %dst[%j] : memref<1048576xf32>
    }

    // expected-remark @below {{load: SINGLE}}
    // expected-remark @below {{full-remat: REJECT_UNSAFE}}
    // expected-remark @below {{partial-remat: REJECT_UNSAFE (reason=intervening-write)}}
    %out = affine.load %dst[%c0] : memref<1048576xf32>
    memref.dealloc %src : memref<4xf32>
    memref.dealloc %dst : memref<1048576xf32>
    return %out : f32
  }
}

// IR unchanged — partial remat rejected.
// CHECK-LABEL: func.func @test
// CHECK:         affine.for
// CHECK:         affine.for
// CHECK:         affine.for
// CHECK:         %[[OUT:.*]] = affine.load %{{.*}}[%c0] : memref<1048576xf32>
// CHECK:         return %[[OUT]] : f32
