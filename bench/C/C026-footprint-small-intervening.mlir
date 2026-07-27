// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-footprint-analysis dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// C026: Footprint — small intervening traffic (256 bytes = 32 x f64).
// Buffer = 1xf64, expensive ALU (sqrt+div, cost=40), 2 consumers.
// Working set = 8 + 256 = 264 bytes → still L1 (< 32KB).
// keepCost = 40+1+2*4 = 49, recomputeCost = 2*40 = 80 → KEEP.

module {
  func.func @footprint_small(%x: f64) -> f64 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f64
    %eps = arith.constant 0.001 : f64

    // expected-remark @+1 {{cost-model: KEEP}}
    %buf = memref.alloc() : memref<1xf64>
    %scratch = memref.alloc() : memref<32xf64>

    %sq = arith.mulf %x, %x : f64
    %sum = arith.addf %sq, %one : f64
    %root = math.sqrt %sum : f64
    %denom = arith.addf %x, %eps : f64
    %val = arith.divf %root, %denom : f64
    memref.store %val, %buf[%c0] : memref<1xf64>

    // Small intervening loop: 32 x 8 bytes = 256 bytes, fits in L1.
    affine.for %i = 0 to 32 {
      memref.store %one, %scratch[%i] : memref<32xf64>
    }

    // expected-remark @below {{cost-model: SKIP_LOAD (buffer kept)}}
    // expected-remark @below {{load: SINGLE}}
    %a = memref.load %buf[%c0] : memref<1xf64>

    // expected-remark @below {{cost-model: SKIP_LOAD (buffer kept)}}
    // expected-remark @below {{load: SINGLE}}
    %b = memref.load %buf[%c0] : memref<1xf64>

    %r = arith.addf %a, %b : f64
    memref.dealloc %scratch : memref<32xf64>
    memref.dealloc %buf : memref<1xf64>
    return %r : f64
  }
}

// CHECK-LABEL: func.func @footprint_small
// CHECK:         math.sqrt
// CHECK:         memref.store
// CHECK:         memref.load
// CHECK:         memref.load
