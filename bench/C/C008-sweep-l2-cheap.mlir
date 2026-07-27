// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// C008: Buffer 128KB (32768xf32) in L2 (load=12), cheap ALU: single addf (cost=1).
// 1 consumer.  keepCost = 1+1+12 = 14, recomputeCost = 1 → RECOMPUTE.

module {
  func.func @sweep_l2_cheap(%x: f32) -> f32 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f32

    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %buf = memref.alloc() : memref<32768xf32>

    %val = arith.addf %x, %one : f32
    memref.store %val, %buf[%c0] : memref<32768xf32>

    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %buf[%c0] : memref<32768xf32>

    memref.dealloc %buf : memref<32768xf32>
    return %v : f32
  }
}

// CHECK-LABEL: func.func @sweep_l2_cheap
// CHECK-NOT:     memref.load
// CHECK:         return
