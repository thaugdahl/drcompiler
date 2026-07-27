// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-footprint-analysis dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// C027: Footprint — intervening traffic evicts buffer from L1.
// Buffer = 1xi32 (4 bytes), cheap ALU (addi, cost=1), 1 consumer.
// Intervening loop writes 8192 x i32 = 32KB, filling L1.
// Working set = 4 + 32768 = 32772 → L2 (load=12).
// keepCost = 1+1+12 = 14, recomputeCost = 1 → RECOMPUTE.

module {
  func.func @footprint_l1_evict(%x: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1 : i32

    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %buf = memref.alloc() : memref<1xi32>
    %other = memref.alloc() : memref<8192xi32>

    %val = arith.addi %x, %one : i32
    memref.store %val, %buf[%c0] : memref<1xi32>

    // Large loop that evicts %buf from L1 (touches 32KB).
    affine.for %i = 0 to 8192 {
      memref.store %one, %other[%i] : memref<8192xi32>
    }

    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %a = memref.load %buf[%c0] : memref<1xi32>

    memref.dealloc %other : memref<8192xi32>
    memref.dealloc %buf : memref<1xi32>
    return %a : i32
  }
}

// CHECK-LABEL: func.func @footprint_l1_evict
// CHECK-NOT:     memref.load %{{.*}} : memref<1xi32>
// CHECK:         return
