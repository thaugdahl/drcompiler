// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// C016: ALU sweep — very expensive.  Buffer 128KB (16384xf64) in L2, load=12.
// ALU = sqrt(20)+sqrt(20)+divf(15) = 55.  2 consumers.
// keepCost = 55+1+24 = 80, recomputeCost = 2*55 = 110 → KEEP.

module {
  func.func @alu_very_expensive(%x: f64) -> f64 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f64

    // expected-remark @+1 {{cost-model: KEEP}}
    %buf = memref.alloc() : memref<16384xf64>

    %s1 = math.sqrt %x : f64
    %s2 = math.sqrt %s1 : f64
    %val = arith.divf %s2, %one : f64
    memref.store %val, %buf[%c0] : memref<16384xf64>

    // expected-remark @below {{cost-model: SKIP_LOAD (buffer kept)}}
    // expected-remark @below {{load: SINGLE}}
    %a = memref.load %buf[%c0] : memref<16384xf64>

    // expected-remark @below {{cost-model: SKIP_LOAD (buffer kept)}}
    // expected-remark @below {{load: SINGLE}}
    %b = memref.load %buf[%c0] : memref<16384xf64>

    %r = arith.addf %a, %b : f64
    memref.dealloc %buf : memref<16384xf64>
    return %r : f64
  }
}

// CHECK-LABEL: func.func @alu_very_expensive
// CHECK:         math.sqrt
// CHECK:         math.sqrt
// CHECK:         memref.store
// CHECK:         memref.load
// CHECK:         memref.load
