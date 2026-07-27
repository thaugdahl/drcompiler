// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-footprint-analysis dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// C030: Footprint — operand memref is cold (not recently accessed).
// Buffer = 1xf64 (8 bytes), ALU = sqrt(20), 2 consumers.
// The stored value is computed from %src, but the intervening loop
// writes to a DIFFERENT memref, so %src is cold.
// Intervening: 8192 x i32 = 32KB → evicts from L1.
// Working set = 8 + 32768 = 32776 → L2 (effectiveLoad=12).
// operandPenalty = (12-4)*1/1 = 8 per consumer.
// recomputeCost = 2*(20+0+8) = 56, keepCost = 20+1+2*12 = 49 → KEEP.

module {
  func.func @footprint_operand_cold(%src: memref<1024xf64>) -> f64 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1 : i32

    // expected-remark @+1 {{cost-model: KEEP}}
    %buf = memref.alloc() : memref<1xf64>
    %other = memref.alloc() : memref<8192xi32>

    %x = memref.load %src[%c0] : memref<1024xf64>
    %val = math.sqrt %x : f64
    memref.store %val, %buf[%c0] : memref<1xf64>

    // Intervening loop writes to %other (NOT %src) → %src is cold.
    affine.for %i = 0 to 8192 {
      memref.store %one, %other[%i] : memref<8192xi32>
    }

    // expected-remark @below {{cost-model: SKIP_LOAD (buffer kept)}}
    // expected-remark @below {{load: SINGLE}}
    %a = memref.load %buf[%c0] : memref<1xf64>

    // expected-remark @below {{cost-model: SKIP_LOAD (buffer kept)}}
    // expected-remark @below {{load: SINGLE}}
    %b = memref.load %buf[%c0] : memref<1xf64>

    %r = arith.addf %a, %b : f64
    memref.dealloc %other : memref<8192xi32>
    memref.dealloc %buf : memref<1xf64>
    return %r : f64
  }
}

// CHECK-LABEL: func.func @footprint_operand_cold
// CHECK:         math.sqrt
// CHECK:         memref.store
// CHECK:         memref.load
// CHECK:         memref.load
