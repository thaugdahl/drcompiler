// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// B008: Full remat — 4-op mul chain stored, loaded elsewhere.

module {
  func.func @remat_mul_chain(%x: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c5 = arith.constant 5 : i32
    %c7 = arith.constant 7 : i32

    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %buf = memref.alloc() : memref<1xi32>

    %m1 = arith.muli %x, %c2 : i32
    %m2 = arith.muli %m1, %c3 : i32
    %m3 = arith.muli %m2, %c5 : i32
    %m4 = arith.muli %m3, %c7 : i32
    memref.store %m4, %buf[%c0] : memref<1xi32>

    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %buf[%c0] : memref<1xi32>
    memref.dealloc %buf : memref<1xi32>
    return %v : i32
  }
}

// CHECK-LABEL: func.func @remat_mul_chain
// CHECK-NOT:     memref.load
// CHECK:         return
