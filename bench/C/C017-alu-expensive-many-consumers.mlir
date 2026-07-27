// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// C017: ALU=20 (sqrt), 4 consumers, buffer 128KB (16384xf64) in L2 (load=12).
// keepCost = 20+1+4*12 = 69, recomputeCost = 4*20 = 80 → KEEP.

module {
  func.func @alu_expensive_many(%x: f64) -> f64 {
    %c0 = arith.constant 0 : index

    // expected-remark @+1 {{cost-model: KEEP}}
    %buf = memref.alloc() : memref<16384xf64>

    %val = math.sqrt %x : f64
    memref.store %val, %buf[%c0] : memref<16384xf64>

    // expected-remark @below {{cost-model: SKIP_LOAD (buffer kept)}}
    // expected-remark @below {{load: SINGLE}}
    %a = memref.load %buf[%c0] : memref<16384xf64>

    // expected-remark @below {{cost-model: SKIP_LOAD (buffer kept)}}
    // expected-remark @below {{load: SINGLE}}
    %b = memref.load %buf[%c0] : memref<16384xf64>

    // expected-remark @below {{cost-model: SKIP_LOAD (buffer kept)}}
    // expected-remark @below {{load: SINGLE}}
    %c = memref.load %buf[%c0] : memref<16384xf64>

    // expected-remark @below {{cost-model: SKIP_LOAD (buffer kept)}}
    // expected-remark @below {{load: SINGLE}}
    %d = memref.load %buf[%c0] : memref<16384xf64>

    %r1 = arith.addf %a, %b : f64
    %r2 = arith.addf %c, %d : f64
    %r = arith.addf %r1, %r2 : f64
    memref.dealloc %buf : memref<16384xf64>
    return %r : f64
  }
}

// CHECK-LABEL: func.func @alu_expensive_many
// CHECK:         math.sqrt
// CHECK:         memref.store
// CHECK:         memref.load
// CHECK:         memref.load
// CHECK:         memref.load
// CHECK:         memref.load
