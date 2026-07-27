// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-footprint-analysis dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// C025: Footprint — no intervening traffic between store and load.
// Buffer is tiny (1xf64 = 8 bytes), expensive ALU (sqrt+div, cost=40),
// 2 consumers.  No intervening ops → storeToLoadFP=0 → effective L1.
// keepCost = 40+1+2*4 = 49, recomputeCost = 2*40 = 80 → KEEP.

module {
  func.func @footprint_none(%x: f64) -> f64 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f64
    %eps = arith.constant 0.001 : f64

    // expected-remark @+1 {{cost-model: KEEP}}
    %buf = memref.alloc() : memref<1xf64>

    %sq = arith.mulf %x, %x : f64
    %sum = arith.addf %sq, %one : f64
    %root = math.sqrt %sum : f64
    %denom = arith.addf %x, %eps : f64
    %val = arith.divf %root, %denom : f64
    memref.store %val, %buf[%c0] : memref<1xf64>

    // expected-remark @below {{cost-model: SKIP_LOAD (buffer kept)}}
    // expected-remark @below {{load: SINGLE}}
    %a = memref.load %buf[%c0] : memref<1xf64>

    // expected-remark @below {{cost-model: SKIP_LOAD (buffer kept)}}
    // expected-remark @below {{load: SINGLE}}
    %b = memref.load %buf[%c0] : memref<1xf64>

    %r = arith.addf %a, %b : f64
    memref.dealloc %buf : memref<1xf64>
    return %r : f64
  }
}

// CHECK-LABEL: func.func @footprint_none
// CHECK:         math.sqrt
// CHECK:         memref.store
// CHECK:         memref.load
// CHECK:         memref.load
