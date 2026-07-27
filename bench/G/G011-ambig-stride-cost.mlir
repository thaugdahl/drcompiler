// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-partial-remat dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// G011: Stride-aware partial remat cost threshold.
// The leaf load uses stride 0 (constant index), so amortized cost is near zero.
// The consumer load is strided (step 16 = cache-line stride).

module {
  func.func @stride_cost() -> f32 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f32

    // expected-remark @below {{cost-model:}}
    %src = memref.alloc() : memref<1048576xf32>
    // expected-remark @below {{cost-model:}}
    %dst = memref.alloc() : memref<1048576xf32>
    // expected-remark @below {{cost-model:}}
    %out_buf = memref.alloc() : memref<65536xf32>

    // Writer to %src with IV-dependent values (breaks chain remat).
    affine.for %i = 0 to 1048576 {
      %ic = arith.index_cast %i : index to i32
      %f = arith.sitofp %ic : i32 to f32
      affine.store %f, %src[%i] : memref<1048576xf32>
    }

    // Store to %dst using constant-index read of %src.
    affine.for %k = 0 to 65536 {
      // expected-remark @below {{full-remat: REJECT_UNSAFE}}
      // expected-remark @below {{load: SINGLE}}
      %vs = affine.load %src[0] : memref<1048576xf32>
      %add = arith.addf %vs, %one : f32
      affine.store %add, %dst[%k * 16] : memref<1048576xf32>
    }

    // Consumer: strided read of %dst into separate out_buf.
    affine.for %j = 0 to 65536 {
      // expected-remark @below {{load: SINGLE}}
      // expected-remark @below {{full-remat: REJECT_UNSAFE}}
      // expected-remark @below {{partial-remat: ACCEPT}}
      %v = affine.load %dst[%j * 16] : memref<1048576xf32>
      affine.store %v, %out_buf[%j] : memref<65536xf32>
    }

    // expected-remark @below {{load: SINGLE}}
    // expected-remark @below {{full-remat: REJECT_UNSAFE}}
    // expected-remark @below {{partial-remat: REJECT_COST}}
    %result = affine.load %out_buf[%c0] : memref<65536xf32>
    memref.dealloc %src : memref<1048576xf32>
    memref.dealloc %dst : memref<1048576xf32>
    memref.dealloc %out_buf : memref<65536xf32>
    return %result : f32
  }
}

// The strided consumer load is replaced by cloned leaf + add.
// CHECK-LABEL: func.func @stride_cost
// CHECK:         affine.for %[[J:.*]] = 0 to 65536 {
// CHECK-NEXT:      affine.load %{{.*}}[0]
// CHECK-NEXT:      arith.addf
