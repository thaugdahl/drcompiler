// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// C002: Buffer exactly 32KB (8192xf32), right at the L1/L2 boundary.
// 32768 bytes <= l1Size=32768, so load latency = 4 (still L1).
// Cheap ALU (addi, cost=1), 1 consumer.
// keepCost = 1+1+4 = 6, recomputeCost = 1 → RECOMPUTE.

module {
  func.func @sweep_l1_boundary(%x: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1 : i32

    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %buf = memref.alloc() : memref<8192xf32>

    %val = arith.addi %x, %one : i32
    %f = arith.sitofp %val : i32 to f32
    memref.store %f, %buf[%c0] : memref<8192xf32>

    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %buf[%c0] : memref<8192xf32>
    %r = arith.fptosi %v : f32 to i32

    memref.dealloc %buf : memref<8192xf32>
    return %r : i32
  }
}

// CHECK-LABEL: func.func @sweep_l1_boundary
// CHECK-NOT:     memref.load
// CHECK:         return
