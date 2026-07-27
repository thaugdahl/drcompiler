// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// C011: ALU sweep — trivial cost.  Buffer 128KB (16384xf64) in L2, load=12.
// ALU = addf(1) = 1.  2 consumers.
// keepCost = 1+1+24 = 26, recomputeCost = 2*1 = 2 → RECOMPUTE.

module {
  func.func @alu_trivial(%x: f64) -> f64 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f64

    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %buf = memref.alloc() : memref<16384xf64>

    %val = arith.addf %x, %one : f64
    memref.store %val, %buf[%c0] : memref<16384xf64>

    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %a = memref.load %buf[%c0] : memref<16384xf64>

    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %b = memref.load %buf[%c0] : memref<16384xf64>

    %r = arith.addf %a, %b : f64
    memref.dealloc %buf : memref<16384xf64>
    return %r : f64
  }
}

// CHECK-LABEL: func.func @alu_trivial
// CHECK-NOT:     memref.load
// CHECK:         return
