// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// C006: Buffer 64MB (16777216xf32), exceeds L2 and no L3 modeled.
// Load latency = 200 (DRAM).  Cheap ALU (addi+sitofp, cost=2), 1 consumer.
// keepCost = 2+1+200 = 203, recomputeCost = 2 → RECOMPUTE.

module {
  func.func @sweep_dram(%x: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1 : i32

    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %buf = memref.alloc() : memref<16777216xf32>

    %val = arith.addi %x, %one : i32
    %f = arith.sitofp %val : i32 to f32
    memref.store %f, %buf[%c0] : memref<16777216xf32>

    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %buf[%c0] : memref<16777216xf32>
    %r = arith.fptosi %v : f32 to i32

    memref.dealloc %buf : memref<16777216xf32>
    return %r : i32
  }
}

// CHECK-LABEL: func.func @sweep_dram
// CHECK-NOT:     memref.load
// CHECK:         return
