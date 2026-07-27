// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-partial-remat dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// C033: Stride-aware partial remat — zero stride (constant index) leaf.
// Same as the reference test: leaf reads src[0] (stride 0 → near-free),
// consumer reads dst[j*16] (stride 16 → full miss at 200 cycles).
// alu=1, leaf~1, load=200 → 1+1 < 200 → ACCEPT.

module {
  func.func @stride_zero() -> f32 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f32

    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %src = memref.alloc() : memref<1048576xf32>
    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %dst = memref.alloc() : memref<1048576xf32>
    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %out = memref.alloc() : memref<65536xf32>

    // Writer: fills src.
    affine.for %i = 0 to 1048576 {
      %i32 = arith.index_cast %i : index to i32
      %f = arith.sitofp %i32 : i32 to f32
      affine.store %f, %src[%i] : memref<1048576xf32>
    }

    // Producer: constant-index read of src[0], stored at stride 16.
    affine.for %k = 0 to 65536 {
      // expected-remark @below {{full-remat: REJECT_UNSAFE}}
      // expected-remark @below {{load: SINGLE}}
      %vs = affine.load %src[0] : memref<1048576xf32>
      %add = arith.addf %vs, %one : f32
      affine.store %add, %dst[%k * 16] : memref<1048576xf32>
    }

    // Consumer: stride-16 read of dst.  Cloned leaf is src[0] → stride 0.
    affine.for %j = 0 to 65536 {
      // expected-remark @below {{load: SINGLE}}
      // expected-remark @below {{full-remat: REJECT_UNSAFE}}
      // expected-remark @below {{partial-remat: ACCEPT}}
      %v = affine.load %dst[%j * 16] : memref<1048576xf32>
      affine.store %v, %out[%j] : memref<65536xf32>
    }

    // expected-remark @below {{load: SINGLE}}
    // expected-remark @below {{full-remat: REJECT_UNSAFE}}
    // expected-remark @below {{partial-remat: REJECT_COST}}
    %result = affine.load %out[%c0] : memref<65536xf32>
    memref.dealloc %src : memref<1048576xf32>
    memref.dealloc %dst : memref<1048576xf32>
    memref.dealloc %out : memref<65536xf32>
    return %result : f32
  }
}

// The strided dst load is replaced by the cloned leaf (src[0]) + addf.
// CHECK-LABEL: func.func @stride_zero
// CHECK:         affine.for %[[J:.*]] = 0 to 65536 {
// CHECK-NEXT:      %[[LS:.*]] = affine.load %{{.*}}[0] : memref<1048576xf32>
// CHECK-NEXT:      %[[ADD:.*]] = arith.addf %[[LS]], %{{.*}} : f32
// CHECK:           affine.store %[[ADD]], %{{.*}}[%[[J]]] : memref<65536xf32>
