// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-footprint-analysis dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// C028: Footprint — intervening traffic evicts buffer from L2.
// Buffer = 1xi32 (4 bytes), cheap ALU (addi, cost=1), 1 consumer.
// Intervening loop writes 65536 x i32 = 256KB, filling L2.
// Working set = 4 + 262144 = 262148 → DRAM (load=200).
// keepCost = 1+1+200 = 202, recomputeCost = 1 → RECOMPUTE.

module {
  func.func @footprint_l2_evict(%x: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1 : i32

    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %buf = memref.alloc() : memref<1xi32>
    %other = memref.alloc() : memref<65536xi32>

    %val = arith.addi %x, %one : i32
    memref.store %val, %buf[%c0] : memref<1xi32>

    // Loop that fills L2 (256KB of traffic).
    affine.for %i = 0 to 65536 {
      memref.store %one, %other[%i] : memref<65536xi32>
    }

    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %a = memref.load %buf[%c0] : memref<1xi32>

    memref.dealloc %other : memref<65536xi32>
    memref.dealloc %buf : memref<1xi32>
    return %a : i32
  }
}

// CHECK-LABEL: func.func @footprint_l2_evict
// CHECK-NOT:     memref.load %{{.*}} : memref<1xi32>
// CHECK:         return
