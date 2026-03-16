// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// Cost model should RECOMPUTE: cheap computation (add), small buffer,
// 1 consumer. Recomputing x+1 (1 cycle) is cheaper than even an L1 load
// (4 cycles).

module {
  func.func @run(%x: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1 : i32

    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %buf = memref.alloc() : memref<1xi32>

    %val = arith.addi %x, %one : i32
    memref.store %val, %buf[%c0] : memref<1xi32>

    // expected-remark @+1 {{load: SINGLE}}
    %v = memref.load %buf[%c0] : memref<1xi32>

    memref.dealloc %buf : memref<1xi32>
    return %v : i32
  }
}

// The load should be replaced — cost model says recompute.
// CHECK-LABEL: func.func @run
// CHECK-NOT:     memref.load
// CHECK:         return
