// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// B003: Direct forward — store before scf.if, load inside branch.

module {
  func.func @direct_cross_block(%x: i32, %cond: i1) -> i32 {
    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %buf = memref.alloc() : memref<i32>
    %c1 = arith.constant 1 : i32
    %val = arith.addi %x, %c1 : i32
    memref.store %val, %buf[] : memref<i32>
    %r = scf.if %cond -> (i32) {
      // expected-remark @below {{direct-forward: ACCEPT}}
      // expected-remark @below {{load: SINGLE}}
      %v = memref.load %buf[] : memref<i32>
      scf.yield %v : i32
    } else {
      %c0 = arith.constant 0 : i32
      scf.yield %c0 : i32
    }
    memref.dealloc %buf : memref<i32>
    return %r : i32
  }
}

// CHECK-LABEL: func.func @direct_cross_block
// CHECK-NOT:     memref.load
// CHECK:         return
