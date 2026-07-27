// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// B016: Full remat — same chain consumed by 4 loads.

module {
  func.func @remat_multi_consumer_chain(%x: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32

    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %buf = memref.alloc() : memref<1xi32>

    %a = arith.addi %x, %c1 : i32
    %b = arith.muli %a, %c2 : i32
    memref.store %b, %buf[%c0] : memref<1xi32>

    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %v1 = memref.load %buf[%c0] : memref<1xi32>
    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %v2 = memref.load %buf[%c0] : memref<1xi32>
    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %v3 = memref.load %buf[%c0] : memref<1xi32>
    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %v4 = memref.load %buf[%c0] : memref<1xi32>

    %s1 = arith.addi %v1, %v2 : i32
    %s2 = arith.addi %v3, %v4 : i32
    %r = arith.addi %s1, %s2 : i32
    memref.dealloc %buf : memref<1xi32>
    return %r : i32
  }
}

// CHECK-LABEL: func.func @remat_multi_consumer_chain
// CHECK-NOT:     memref.load
// CHECK:         return
