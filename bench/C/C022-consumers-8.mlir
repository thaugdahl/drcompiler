// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// C022: Consumer count sweep — 8 consumers.  ALU=20 (sqrt), buffer 128KB L2.
// keepCost = 20+1+96 = 117, recomputeCost = 160 → KEEP.

module {
  func.func @consumers_8(%x: f64) -> f64 {
    %c0 = arith.constant 0 : index

    // expected-remark @+1 {{cost-model: KEEP}}
    %buf = memref.alloc() : memref<16384xf64>

    %val = math.sqrt %x : f64
    memref.store %val, %buf[%c0] : memref<16384xf64>

    // expected-remark @below {{cost-model: SKIP_LOAD (buffer kept)}}
    // expected-remark @below {{load: SINGLE}}
    %v1 = memref.load %buf[%c0] : memref<16384xf64>
    // expected-remark @below {{cost-model: SKIP_LOAD (buffer kept)}}
    // expected-remark @below {{load: SINGLE}}
    %v2 = memref.load %buf[%c0] : memref<16384xf64>
    // expected-remark @below {{cost-model: SKIP_LOAD (buffer kept)}}
    // expected-remark @below {{load: SINGLE}}
    %v3 = memref.load %buf[%c0] : memref<16384xf64>
    // expected-remark @below {{cost-model: SKIP_LOAD (buffer kept)}}
    // expected-remark @below {{load: SINGLE}}
    %v4 = memref.load %buf[%c0] : memref<16384xf64>
    // expected-remark @below {{cost-model: SKIP_LOAD (buffer kept)}}
    // expected-remark @below {{load: SINGLE}}
    %v5 = memref.load %buf[%c0] : memref<16384xf64>
    // expected-remark @below {{cost-model: SKIP_LOAD (buffer kept)}}
    // expected-remark @below {{load: SINGLE}}
    %v6 = memref.load %buf[%c0] : memref<16384xf64>
    // expected-remark @below {{cost-model: SKIP_LOAD (buffer kept)}}
    // expected-remark @below {{load: SINGLE}}
    %v7 = memref.load %buf[%c0] : memref<16384xf64>
    // expected-remark @below {{cost-model: SKIP_LOAD (buffer kept)}}
    // expected-remark @below {{load: SINGLE}}
    %v8 = memref.load %buf[%c0] : memref<16384xf64>

    %r1 = arith.addf %v1, %v2 : f64
    %r2 = arith.addf %v3, %v4 : f64
    %r3 = arith.addf %v5, %v6 : f64
    %r4 = arith.addf %v7, %v8 : f64
    %r5 = arith.addf %r1, %r2 : f64
    %r6 = arith.addf %r3, %r4 : f64
    %r = arith.addf %r5, %r6 : f64
    memref.dealloc %buf : memref<16384xf64>
    return %r : f64
  }
}

// CHECK-LABEL: func.func @consumers_8
// CHECK:         math.sqrt
// CHECK:         memref.store
// CHECK:         memref.load
