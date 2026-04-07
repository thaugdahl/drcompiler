// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-footprint-analysis dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// Footprint analysis should detect that the large loop between store and load
// evicts the small buffer from L1, pushing effective load latency higher.
// With cheap computation (addi, ~1 cycle) and an evicted buffer, the cost
// model should favor RECOMPUTE.
//
// Without footprint analysis the buffer is 4 bytes (fits in L1, 4-cycle load)
// and KEEP would win with 2 consumers.  With footprint analysis, the
// intervening loop touches 32KB (8192 x i32 = 32768 bytes), filling L1 and
// evicting the small buffer.

module {
  func.func @run(%x: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1 : i32

    // expected-remark @below {{cost-model: RECOMPUTE}}
    %buf = memref.alloc() : memref<1xi32>
    %other = memref.alloc() : memref<8192xi32>

    %val = arith.addi %x, %one : i32
    memref.store %val, %buf[%c0] : memref<1xi32>

    // Large loop that evicts %buf from L1 (touches 32KB of %other).
    affine.for %i = 0 to 8192 {
      memref.store %one, %other[%i] : memref<8192xi32>
    }

    // expected-remark @below {{load: SINGLE}}
    %a = memref.load %buf[%c0] : memref<1xi32>
    // expected-remark @below {{load: SINGLE}}
    %b = memref.load %buf[%c0] : memref<1xi32>
    %r = arith.addi %a, %b : i32

    memref.dealloc %other : memref<8192xi32>
    memref.dealloc %buf : memref<1xi32>
    return %r : i32
  }
}

// The loads should be replaced -- footprint-aware cost model says RECOMPUTE.
// CHECK-LABEL: func.func @run
// CHECK-NOT:     memref.load %buf
// CHECK:         return
