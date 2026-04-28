// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-footprint-analysis dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// Operand warmth test: the stored value is computed from a load of %src.
// The intervening loop between store and load also reads from %src,
// so the operand memref is "warm" (recently accessed, likely in cache).
// This means the operand penalty for recomputation should be 0,
// making recomputation cheap despite the large intervening footprint.
//
// val = src[0] + 1   (cheap, ~1 cycle ALU)
// store val -> buf[0]
// loop: read src[i] for i=0..8191  (touches 32KB of src -- evicts buf)
// load buf[0]   (buf evicted, but src is warm from the loop)
//
// Without warm-operand analysis, the penalty would be nonzero and might
// disfavor recomputation.  With it, recompute is clearly profitable.

module {
  func.func @run(%src: memref<8192xi32>) -> i32 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1 : i32

    // expected-remark @below {{cost-model: RECOMPUTE}}
    %buf = memref.alloc() : memref<1xi32>

    %x = memref.load %src[%c0] : memref<8192xi32>
    %val = arith.addi %x, %one : i32
    memref.store %val, %buf[%c0] : memref<1xi32>

    // Large loop that reads from the SAME memref (%src) the operand came from.
    // This keeps %src warm in cache even though it evicts %buf.
    %sum = arith.constant 0 : i32
    %r = affine.for %i = 0 to 8192 iter_args(%acc = %sum) -> (i32) {
      %v = memref.load %src[%i] : memref<8192xi32>
      %next = arith.addi %acc, %v : i32
      affine.yield %next : i32
    }

    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %a = memref.load %buf[%c0] : memref<1xi32>
    %result = arith.addi %a, %r : i32

    memref.dealloc %buf : memref<1xi32>
    return %result : i32
  }
}

// The load should be replaced — operand is warm, recompute is cheap.
// CHECK-LABEL: func.func @run
// CHECK-NOT:     memref.load %buf
// CHECK:         return
