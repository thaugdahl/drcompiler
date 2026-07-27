// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// C014: ALU sweep — break-even point.  Buffer 128KB (16384xf64) in L2, load=12.
// ALU = sqrt(20)+addf(1)*5 = 25.  2 consumers.
// keepCost = 25+1+24 = 50, recomputeCost = 2*25 = 50.
// Tie: recompute <= keep (50<=50) → RECOMPUTE.

module {
  func.func @alu_break_even(%x: f64) -> f64 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f64

    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %buf = memref.alloc() : memref<16384xf64>

    %r0 = math.sqrt %x : f64
    %r1 = arith.addf %r0, %one : f64
    %r2 = arith.addf %r1, %one : f64
    %r3 = arith.addf %r2, %one : f64
    %r4 = arith.addf %r3, %one : f64
    %val = arith.addf %r4, %one : f64
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

// CHECK-LABEL: func.func @alu_break_even
// CHECK-NOT:     memref.load
// CHECK:         return
