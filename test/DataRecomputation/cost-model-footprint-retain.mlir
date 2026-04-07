// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-footprint-analysis dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// Footprint analysis with minimal intervening traffic.
// The buffer is small (8 bytes), computation is expensive (sqrt + div),
// and the intervening loop only touches 32 bytes (4 x f64), which stays
// well within L1.  The cost model should KEEP the buffer.

module {
  func.func @run(%x: f64) -> f64 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f64
    %eps = arith.constant 0.001 : f64

    // expected-remark @below {{cost-model: KEEP}}
    %buf = memref.alloc() : memref<1xf64>
    %scratch = memref.alloc() : memref<4xf64>

    // Expensive computation: sqrt(x^2 + 1) / (x + eps)
    %sq = arith.mulf %x, %x : f64
    %sum = arith.addf %sq, %one : f64
    %root = math.sqrt %sum : f64
    %denom = arith.addf %x, %eps : f64
    %val = arith.divf %root, %denom : f64
    memref.store %val, %buf[%c0] : memref<1xf64>

    // Tiny loop: only 4 iterations x 8 bytes = 32 bytes. No eviction.
    affine.for %i = 0 to 4 {
      memref.store %one, %scratch[%i] : memref<4xf64>
    }

    // Consumer 1
    // expected-remark @below {{load: SINGLE}}
    %a = memref.load %buf[%c0] : memref<1xf64>

    // Consumer 2
    // expected-remark @below {{load: SINGLE}}
    %b = memref.load %buf[%c0] : memref<1xf64>

    %r = arith.addf %a, %b : f64
    memref.dealloc %scratch : memref<4xf64>
    memref.dealloc %buf : memref<1xf64>
    return %r : f64
  }
}

// The loads should survive -- cost model says KEEP.
// CHECK-LABEL: func.func @run
// CHECK:         math.sqrt
// CHECK:         memref.store
// CHECK:         memref.load
// CHECK:         memref.load
