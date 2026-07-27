// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// C019: Consumer count sweep — 1 consumer.  ALU=20 (sqrt), buffer 128KB L2.
// keepCost = 20+1+12 = 33, recomputeCost = 20 → RECOMPUTE.

module {
  func.func @consumers_1(%x: f64) -> f64 {
    %c0 = arith.constant 0 : index

    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %buf = memref.alloc() : memref<16384xf64>

    %val = math.sqrt %x : f64
    memref.store %val, %buf[%c0] : memref<16384xf64>

    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %a = memref.load %buf[%c0] : memref<16384xf64>

    memref.dealloc %buf : memref<16384xf64>
    return %a : f64
  }
}

// CHECK-LABEL: func.func @consumers_1
// CHECK-NOT:     memref.load
// CHECK:         return
