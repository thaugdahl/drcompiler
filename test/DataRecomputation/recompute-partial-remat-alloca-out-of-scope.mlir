// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-partial-remat dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// Partial-remat safety check rejects leaf loads whose memref is a
// memref.alloca scoped inside a region that does not encompass the
// insertion point (mirrors MNIST cases #1–#3 where scalar
// accumulators are scoped to an inner loop).

module {
  func.func @test() -> f32 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f32
    %zero = arith.constant 0.0 : f32
    // expected-remark @+1 {{cost-model:}}
    %dst = memref.alloc() : memref<1048576xf32>

    affine.for %j = 0 to 1048576 {
      // Scalar accumulator scoped to the outer loop body — init + body
      // stores reach %final with SINGLE provenance but writeCount=2.
      // expected-remark @+1 {{cost-model:}}
      %acc = memref.alloca() : memref<f32>
      memref.store %zero, %acc[] : memref<f32>
      affine.for %k = 0 to 4 {
        // expected-remark @+1 {{load: MULTI}}
        %cur = memref.load %acc[] : memref<f32>
        %new = arith.addf %cur, %one : f32
        memref.store %new, %acc[] : memref<f32>
      }
      // expected-remark @+2 {{load: SINGLE}}
      // expected-remark @+1 {{partial-remat: REJECT_UNSAFE}}
      %final = memref.load %acc[] : memref<f32>
      %s = arith.addf %final, %one : f32
      affine.store %s, %dst[%j] : memref<1048576xf32>
    }

    // expected-remark @+2 {{load: SINGLE}}
    // expected-remark @+1 {{partial-remat: REJECT_UNSAFE (reason=alloca-out-of-scope)}}
    %out = affine.load %dst[%c0] : memref<1048576xf32>
    memref.dealloc %dst : memref<1048576xf32>
    return %out : f32
  }
}

// IR unchanged — partial remat rejected.
// CHECK-LABEL: func.func @test
// CHECK:         affine.for
// CHECK:         %[[OUT:.*]] = affine.load %{{.*}}[%c0] : memref<1048576xf32>
// CHECK:         return %[[OUT]] : f32
