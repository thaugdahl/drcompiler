// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// C010: Degenerate buffer — memref<1xf32> = 4 bytes, trivially L1.
// Cheap ALU (addf, cost=1), 1 consumer.
// keepCost = 1+1+4 = 6, recomputeCost = 1 → RECOMPUTE.

module {
  func.func @sweep_zero_size(%x: f32) -> f32 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f32

    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %buf = memref.alloc() : memref<1xf32>

    %val = arith.addf %x, %one : f32
    memref.store %val, %buf[%c0] : memref<1xf32>

    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %buf[%c0] : memref<1xf32>

    memref.dealloc %buf : memref<1xf32>
    return %v : f32
  }
}

// CHECK-LABEL: func.func @sweep_zero_size
// CHECK-NOT:     memref.load
// CHECK:         return
