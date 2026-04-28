// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-footprint-analysis dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// Nested loop footprint test: a 64x64 nested loop writing i32 (4 bytes)
// touches 64*64*4 = 16384 bytes per iteration of the outer loop...
// but the total footprint is 64*64*4 = 16384 bytes (fits in L1 = 32KB).
// With an expensive computation (sqrt + div), the cost model should KEEP.

module {
  func.func @run(%x: f64) -> f64 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f64
    %eps = arith.constant 0.001 : f64
    %ione = arith.constant 1 : i32

    // expected-remark @below {{cost-model: KEEP}}
    %buf = memref.alloc() : memref<1xf64>
    %mat = memref.alloc() : memref<64x64xi32>

    // Expensive computation
    %sq = arith.mulf %x, %x : f64
    %sum = arith.addf %sq, %one : f64
    %root = math.sqrt %sum : f64
    %denom = arith.addf %x, %eps : f64
    %val = arith.divf %root, %denom : f64
    memref.store %val, %buf[%c0] : memref<1xf64>

    // Nested loop: 64 x 64 x 4 bytes = 16384 bytes, fits in L1
    affine.for %i = 0 to 64 {
      affine.for %j = 0 to 64 {
        memref.store %ione, %mat[%i, %j] : memref<64x64xi32>
      }
    }

    // expected-remark @below {{cost-model: SKIP_LOAD (buffer kept)}}
    // expected-remark @below {{load: SINGLE}}
    %a = memref.load %buf[%c0] : memref<1xf64>
    // expected-remark @below {{cost-model: SKIP_LOAD (buffer kept)}}
    // expected-remark @below {{load: SINGLE}}
    %b = memref.load %buf[%c0] : memref<1xf64>

    %r = arith.addf %a, %b : f64
    memref.dealloc %mat : memref<64x64xi32>
    memref.dealloc %buf : memref<1xf64>
    return %r : f64
  }
}

// The loads should survive -- nested loop fits in L1, expensive computation.
// CHECK-LABEL: func.func @run
// CHECK:         math.sqrt
// CHECK:         memref.store
// CHECK:         memref.load
// CHECK:         memref.load
