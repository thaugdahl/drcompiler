// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// C007: Buffer 16KB (2048xf64) fits in L1 (load=4), but expensive ALU:
// sqrt(x*x+1)/(x+eps) → mulf(3)+addf(1)+sqrt(20)+addf(1)+divf(15) = 40.
// 2 consumers.  keepCost = 40+1+2*4 = 49, recomputeCost = 2*40 = 80 → KEEP.

module {
  func.func @sweep_l1_expensive(%x: f64) -> f64 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f64
    %eps = arith.constant 0.001 : f64

    // expected-remark @+1 {{cost-model: KEEP}}
    %buf = memref.alloc() : memref<2048xf64>

    %sq = arith.mulf %x, %x : f64
    %sum = arith.addf %sq, %one : f64
    %root = math.sqrt %sum : f64
    %denom = arith.addf %x, %eps : f64
    %val = arith.divf %root, %denom : f64
    memref.store %val, %buf[%c0] : memref<2048xf64>

    // expected-remark @below {{cost-model: SKIP_LOAD (buffer kept)}}
    // expected-remark @below {{load: SINGLE}}
    %a = memref.load %buf[%c0] : memref<2048xf64>

    // expected-remark @below {{cost-model: SKIP_LOAD (buffer kept)}}
    // expected-remark @below {{load: SINGLE}}
    %b = memref.load %buf[%c0] : memref<2048xf64>

    %r = arith.addf %a, %b : f64
    memref.dealloc %buf : memref<2048xf64>
    return %r : f64
  }
}

// CHECK-LABEL: func.func @sweep_l1_expensive
// CHECK:         math.sqrt
// CHECK:         memref.store
// CHECK:         memref.load
// CHECK:         memref.load
