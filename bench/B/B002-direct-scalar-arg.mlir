// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// B002: Direct forward — store function arg, load same block.

module {
  func.func @direct_scalar_arg(%x: i32) -> i32 {
    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %buf = memref.alloc() : memref<i32>
    memref.store %x, %buf[] : memref<i32>
    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %buf[] : memref<i32>
    memref.dealloc %buf : memref<i32>
    return %v : i32
  }
}

// CHECK-LABEL: func.func @direct_scalar_arg
// CHECK-SAME:    %[[X:.*]]: i32
// CHECK-NOT:     memref.load
// CHECK:         return %[[X]] : i32
