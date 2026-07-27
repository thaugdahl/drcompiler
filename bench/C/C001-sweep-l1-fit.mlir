// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// C001: Buffer 16KB (4096xf32) fits in L1.  Cheap ALU (addi, cost=1),
// 1 consumer.  keepCost = 1+1+1*4 = 6, recomputeCost = 1*1 = 1.
// Recompute wins easily.

module {
  func.func @sweep_l1_fit(%x: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1 : i32

    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %buf = memref.alloc() : memref<4096xf32>

    %val = arith.addi %x, %one : i32
    %f = arith.sitofp %val : i32 to f32
    memref.store %f, %buf[%c0] : memref<4096xf32>

    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %buf[%c0] : memref<4096xf32>
    %r = arith.fptosi %v : f32 to i32

    memref.dealloc %buf : memref<4096xf32>
    return %r : i32
  }
}

// CHECK-LABEL: func.func @sweep_l1_fit
// CHECK-NOT:     memref.load
// CHECK:         return
