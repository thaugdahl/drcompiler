// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-partial-remat dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// Cost gate rejects partial rematerialization when recomputation cost
// (alu + leaf-load latency) is not strictly cheaper than the original
// load. Both buffers are small (L1-resident, 4 cycles): the addf + L1
// leaf load ties the L1 load, so partial remat is not profitable.

module {
  func.func @test() -> f32 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f32
    // expected-remark @+1 {{cost-model:}}
    %src = memref.alloc() : memref<4xf32>
    // expected-remark @+1 {{cost-model:}}
    %dst = memref.alloc() : memref<4xf32>

    affine.for %i = 0 to 4 {
      %i32 = arith.index_cast %i : index to i32
      %f = arith.sitofp %i32 : i32 to f32
      affine.store %f, %src[%i] : memref<4xf32>
    }

    affine.for %j = 0 to 4 {
      // expected-remark @+1 {{load: SINGLE}}
      %v = affine.load %src[0] : memref<4xf32>
      %s = arith.addf %v, %one : f32
      affine.store %s, %dst[%j] : memref<4xf32>
    }

    // expected-remark @+2 {{load: SINGLE}}
    // expected-remark @+1 {{partial-remat: REJECT_COST}}
    %out = affine.load %dst[%c0] : memref<4xf32>
    memref.dealloc %src : memref<4xf32>
    memref.dealloc %dst : memref<4xf32>
    return %out : f32
  }
}

// IR unchanged — the %out load survives.
// CHECK-LABEL: func.func @test
// CHECK:         affine.for
// CHECK:         affine.for
// CHECK:         %[[OUT:.*]] = affine.load %{{.*}}[%c0] : memref<4xf32>
// CHECK:         return %[[OUT]] : f32
