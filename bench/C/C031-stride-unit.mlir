// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-partial-remat dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// C031: Stride-aware partial remat — unit stride (step=1).
// Leaf load at stride 1 in the consumer loop amortizes across the cache
// line.  Consumer load at stride 16 pays a full miss.
// The leaf cost is low enough for partial-remat to accept.

module {
  func.func @stride_unit() -> f32 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f32

    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %src = memref.alloc() : memref<1048576xf32>
    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %dst = memref.alloc() : memref<1048576xf32>
    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %out = memref.alloc() : memref<65536xf32>

    // Writer: fills src with loop-dependent data.
    affine.for %i = 0 to 1048576 {
      %i32 = arith.index_cast %i : index to i32
      %f = arith.sitofp %i32 : i32 to f32
      affine.store %f, %src[%i] : memref<1048576xf32>
    }

    // Producer: unit-stride read of src, stored with large stride into dst.
    affine.for %k = 0 to 65536 {
      // expected-remark @below {{full-remat: REJECT_UNSAFE}}
      // expected-remark @below {{load: SINGLE}}
      %vs = affine.load %src[%k] : memref<1048576xf32>
      %add = arith.addf %vs, %one : f32
      affine.store %add, %dst[%k * 16] : memref<1048576xf32>
    }

    // Consumer: large-stride read of dst.  Partial-remat clones the
    // unit-stride leaf (src[j]) → amortized cost is low.
    affine.for %j = 0 to 65536 {
      // expected-remark @below {{load: SINGLE}}
      // expected-remark @below {{full-remat: REJECT_UNSAFE}}
      // expected-remark @below {{partial-remat: ACCEPT}}
      %v = affine.load %dst[%j * 16] : memref<1048576xf32>
      affine.store %v, %out[%j] : memref<65536xf32>
    }

    // expected-remark @below {{load: SINGLE}}
    // expected-remark @below {{full-remat: REJECT_UNSAFE}}
    // expected-remark @below {{partial-remat: REJECT_UNSAFE}}
    %result = affine.load %out[%c0] : memref<65536xf32>
    memref.dealloc %src : memref<1048576xf32>
    memref.dealloc %dst : memref<1048576xf32>
    memref.dealloc %out : memref<65536xf32>
    return %result : f32
  }
}

// The strided dst load is replaced by a clone of the leaf + addf.
// CHECK-LABEL: func.func @stride_unit
// CHECK:         affine.for
// CHECK:           affine.load %{{.*}}[%{{.*}}] : memref<1048576xf32>
// CHECK:           arith.addf
// CHECK:           affine.store
