// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// B005: Direct forward — one store, 4 loads (multi-consumer).

module {
  func.func @direct_multi_consumer(%x: i32) -> i32 {
    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %buf = memref.alloc() : memref<i32>
    %c1 = arith.constant 1 : i32
    %val = arith.addi %x, %c1 : i32
    memref.store %val, %buf[] : memref<i32>

    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %a = memref.load %buf[] : memref<i32>
    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %b = memref.load %buf[] : memref<i32>
    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %c = memref.load %buf[] : memref<i32>
    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %d = memref.load %buf[] : memref<i32>

    %s1 = arith.addi %a, %b : i32
    %s2 = arith.addi %c, %d : i32
    %r = arith.addi %s1, %s2 : i32
    memref.dealloc %buf : memref<i32>
    return %r : i32
  }
}

// CHECK-LABEL: func.func @direct_multi_consumer
// CHECK-NOT:     memref.load
// CHECK:         return
