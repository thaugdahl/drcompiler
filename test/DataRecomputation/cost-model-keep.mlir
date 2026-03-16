// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// Cost model should KEEP the buffer: expensive computation (sqrt + div),
// small buffer (8 bytes, fits in L1), 2 consumer loads.
// Recomputing sqrt+div twice costs more than 2 L1 loads.

module {
  func.func @run(%x: f64) -> f64 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f64
    %eps = arith.constant 0.001 : f64

    // expected-remark @+1 {{cost-model: KEEP}}
    %buf = memref.alloc() : memref<1xf64>

    // Expensive computation: sqrt(x^2 + 1) / (x + eps)
    %sq = arith.mulf %x, %x : f64
    %sum = arith.addf %sq, %one : f64
    %root = math.sqrt %sum : f64
    %denom = arith.addf %x, %eps : f64
    %val = arith.divf %root, %denom : f64
    memref.store %val, %buf[%c0] : memref<1xf64>

    // Consumer 1
    // expected-remark @+1 {{load: SINGLE}}
    %a = memref.load %buf[%c0] : memref<1xf64>

    // Consumer 2
    // expected-remark @+1 {{load: SINGLE}}
    %b = memref.load %buf[%c0] : memref<1xf64>

    %r = arith.addf %a, %b : f64
    memref.dealloc %buf : memref<1xf64>
    return %r : f64
  }
}

// The loads should survive — cost model says keep the buffer.
// CHECK-LABEL: func.func @run
// CHECK:         math.sqrt
// CHECK:         memref.store
// CHECK:         memref.load
// CHECK:         memref.load
