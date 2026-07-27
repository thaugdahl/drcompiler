// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// C015: ALU sweep — expensive.  Buffer 128KB (16384xf64) in L2, load=12.
// ALU = sqrt(20)+divf(15) = 35.  2 consumers.
// keepCost = 35+1+24 = 60, recomputeCost = 2*35 = 70 → KEEP.

module {
  func.func @alu_expensive(%x: f64) -> f64 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f64

    // expected-remark @+1 {{cost-model: KEEP}}
    %buf = memref.alloc() : memref<16384xf64>

    %r0 = math.sqrt %x : f64
    %val = arith.divf %r0, %one : f64
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

// CHECK-LABEL: func.func @alu_expensive
// CHECK:         math.sqrt
// CHECK:         memref.store
// CHECK:         memref.load
// CHECK:         memref.load
