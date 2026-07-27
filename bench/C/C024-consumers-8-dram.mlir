// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// C024: 8 consumers, ALU=20 (sqrt), buffer 64MB (8388608xf64) → DRAM (load=200).
// keepCost = 20+1+1600 = 1621, recomputeCost = 160 → RECOMPUTE.
// Many consumers, but DRAM latency dominates.

module {
  func.func @consumers_8_dram(%x: f64) -> f64 {
    %c0 = arith.constant 0 : index

    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %buf = memref.alloc() : memref<8388608xf64>

    %val = math.sqrt %x : f64
    memref.store %val, %buf[%c0] : memref<8388608xf64>

    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %v1 = memref.load %buf[%c0] : memref<8388608xf64>
    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %v2 = memref.load %buf[%c0] : memref<8388608xf64>
    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %v3 = memref.load %buf[%c0] : memref<8388608xf64>
    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %v4 = memref.load %buf[%c0] : memref<8388608xf64>
    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %v5 = memref.load %buf[%c0] : memref<8388608xf64>
    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %v6 = memref.load %buf[%c0] : memref<8388608xf64>
    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %v7 = memref.load %buf[%c0] : memref<8388608xf64>
    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %v8 = memref.load %buf[%c0] : memref<8388608xf64>

    %r1 = arith.addf %v1, %v2 : f64
    %r2 = arith.addf %v3, %v4 : f64
    %r3 = arith.addf %v5, %v6 : f64
    %r4 = arith.addf %v7, %v8 : f64
    %r5 = arith.addf %r1, %r2 : f64
    %r6 = arith.addf %r3, %r4 : f64
    %r = arith.addf %r5, %r6 : f64
    memref.dealloc %buf : memref<8388608xf64>
    return %r : f64
  }
}

// CHECK-LABEL: func.func @consumers_8_dram
// CHECK-NOT:     memref.load
// CHECK:         return
