// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// B007: Full remat — 4-op add chain stored, loaded elsewhere.

module {
  func.func @remat_add_chain(%x: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c4 = arith.constant 4 : i32

    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %buf = memref.alloc() : memref<1xi32>

    %a1 = arith.addi %x, %c1 : i32
    %a2 = arith.addi %a1, %c2 : i32
    %a3 = arith.addi %a2, %c3 : i32
    %a4 = arith.addi %a3, %c4 : i32
    memref.store %a4, %buf[%c0] : memref<1xi32>

    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %buf[%c0] : memref<1xi32>
    memref.dealloc %buf : memref<1xi32>
    return %v : i32
  }
}

// CHECK-LABEL: func.func @remat_add_chain
// CHECK-NOT:     memref.load
// CHECK:         return
