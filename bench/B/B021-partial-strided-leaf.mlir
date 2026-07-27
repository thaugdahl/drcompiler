// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-partial-remat dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// B021: Partial remat with strided leaf.
// The leaf load uses constant index (stride 0 in consumer loop), so its
// cost amortizes to near zero. The consumer load is strided.

module {
  func.func @partial_strided_leaf() -> f32 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f32
    // expected-remark @+1 {{cost-model:}}
    %src = memref.alloc() : memref<1048576xf32>
    // expected-remark @+1 {{cost-model:}}
    %dst = memref.alloc() : memref<1048576xf32>
    // expected-remark @+1 {{cost-model:}}
    %out_buf = memref.alloc() : memref<65536xf32>

    affine.for %i = 0 to 1048576 {
      %i32 = arith.index_cast %i : index to i32
      %f = arith.sitofp %i32 : i32 to f32
      affine.store %f, %src[%i] : memref<1048576xf32>
    }

    affine.for %k = 0 to 65536 {
      // expected-remark @below {{full-remat: REJECT_UNSAFE}}
      // expected-remark @below {{load: SINGLE}}
      %vs = affine.load %src[0] : memref<1048576xf32>
      %add = arith.addf %vs, %one : f32
      affine.store %add, %dst[%k * 16] : memref<1048576xf32>
    }

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

// Inside the consumer loop the strided dst load is replaced.
// CHECK-LABEL: func.func @partial_strided_leaf
// CHECK:         affine.for %[[J:.*]] = 0 to 65536 {
// CHECK-NEXT:      %[[LS:.*]] = affine.load %{{.*}}[0] : memref<1048576xf32>
// CHECK-NEXT:      %[[ADD:.*]] = arith.addf %[[LS]], %{{.*}} : f32
// CHECK:           affine.store %[[ADD]], %{{.*}}[%[[J]]] : memref<65536xf32>
// CHECK:         }
