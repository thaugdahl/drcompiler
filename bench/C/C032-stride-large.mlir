// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-partial-remat dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// C032: Stride-aware partial remat — large stride on BOTH leaf and consumer.
// Leaf at stride 16 * 4B = 64B+1 → full miss per iter.
// Consumer at stride 16 → also full miss per iter.
// But leaf index is from producer loop (k), not consumer loop (j), so at the
// consumer site the leaf is effectively stride-0 → amortized near-free.
// Result: ACCEPT (same as C031 — stride matters only at the insertion site).

module {
  func.func @stride_large() -> f32 {
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

    // Producer: large-stride read of src, stored with large stride into dst.
    affine.for %k = 0 to 65536 {
      // expected-remark @below {{full-remat: REJECT_UNSAFE}}
      // expected-remark @below {{load: SINGLE}}
      %vs = affine.load %src[%k * 16] : memref<1048576xf32>
      %add = arith.addf %vs, %one : f32
      affine.store %add, %dst[%k * 16] : memref<1048576xf32>
    }

    // Consumer: large-stride read of dst.  The cloned leaf's index is
    // from producer loop (not consumer's j), so effectively stride 0.
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

// The strided dst load is replaced by the cloned leaf + addf.
// CHECK-LABEL: func.func @stride_large
// CHECK:         affine.for %{{.*}} = 0 to 65536 {
// CHECK:           affine.load %{{.*}}[%{{.*}} * 16] : memref<1048576xf32>
// CHECK:           arith.addf
// CHECK:           affine.store
