// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// B009: Full remat — add+mul+sub chain (6 ops).

module {
  func.func @remat_mixed_arith(%x: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32

    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %buf = memref.alloc() : memref<1xi32>

    %a = arith.addi %x, %c1 : i32
    %b = arith.muli %a, %c2 : i32
    %c = arith.subi %b, %c3 : i32
    %d = arith.addi %c, %c1 : i32
    %e = arith.muli %d, %c2 : i32
    %f = arith.subi %e, %c3 : i32
    memref.store %f, %buf[%c0] : memref<1xi32>

    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %buf[%c0] : memref<1xi32>
    memref.dealloc %buf : memref<1xi32>
    return %v : i32
  }
}

// CHECK-LABEL: func.func @remat_mixed_arith
// CHECK-NOT:     memref.load
// CHECK:         return
