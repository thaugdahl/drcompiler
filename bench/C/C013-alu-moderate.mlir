// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// C013: ALU sweep — moderate cost.  Buffer 128KB (16384xf64) in L2, load=12.
// ALU = mulf(3)+addf(1)+mulf(3)+addf(1)+addf(1)+addf(1) = 10.  2 consumers.
// keepCost = 10+1+24 = 35, recomputeCost = 2*10 = 20 → RECOMPUTE.

module {
  func.func @alu_moderate(%x: f64) -> f64 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f64

    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %buf = memref.alloc() : memref<16384xf64>

    %sq = arith.mulf %x, %x : f64
    %t1 = arith.addf %sq, %one : f64
    %t2 = arith.mulf %t1, %x : f64
    %t3 = arith.addf %t2, %one : f64
    %t4 = arith.addf %t3, %one : f64
    %val = arith.addf %t4, %one : f64
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

// CHECK-LABEL: func.func @alu_moderate
// CHECK-NOT:     memref.load
// CHECK:         return
