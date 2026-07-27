// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// B001: Direct forward — store constant, load same block.

module {
  func.func @direct_scalar_const() -> i32 {
    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %buf = memref.alloc() : memref<i32>
    %c42 = arith.constant 42 : i32
    memref.store %c42, %buf[] : memref<i32>
    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %buf[] : memref<i32>
    memref.dealloc %buf : memref<i32>
    return %v : i32
  }
}

// CHECK-LABEL: func.func @direct_scalar_const
// CHECK:         %[[C42:.*]] = arith.constant 42 : i32
// CHECK-NOT:     memref.load
// CHECK:         return %[[C42]] : i32
