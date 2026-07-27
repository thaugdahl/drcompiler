// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// B014: Full remat — 32-op linear chain.

module {
  func.func @remat_long_chain(%x: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : i32

    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %buf = memref.alloc() : memref<1xi32>

    %a01 = arith.addi %x,   %c1 : i32
    %a02 = arith.addi %a01, %c1 : i32
    %a03 = arith.addi %a02, %c1 : i32
    %a04 = arith.addi %a03, %c1 : i32
    %a05 = arith.addi %a04, %c1 : i32
    %a06 = arith.addi %a05, %c1 : i32
    %a07 = arith.addi %a06, %c1 : i32
    %a08 = arith.addi %a07, %c1 : i32
    %a09 = arith.addi %a08, %c1 : i32
    %a10 = arith.addi %a09, %c1 : i32
    %a11 = arith.addi %a10, %c1 : i32
    %a12 = arith.addi %a11, %c1 : i32
    %a13 = arith.addi %a12, %c1 : i32
    %a14 = arith.addi %a13, %c1 : i32
    %a15 = arith.addi %a14, %c1 : i32
    %a16 = arith.addi %a15, %c1 : i32
    %a17 = arith.addi %a16, %c1 : i32
    %a18 = arith.addi %a17, %c1 : i32
    %a19 = arith.addi %a18, %c1 : i32
    %a20 = arith.addi %a19, %c1 : i32
    %a21 = arith.addi %a20, %c1 : i32
    %a22 = arith.addi %a21, %c1 : i32
    %a23 = arith.addi %a22, %c1 : i32
    %a24 = arith.addi %a23, %c1 : i32
    %a25 = arith.addi %a24, %c1 : i32
    %a26 = arith.addi %a25, %c1 : i32
    %a27 = arith.addi %a26, %c1 : i32
    %a28 = arith.addi %a27, %c1 : i32
    %a29 = arith.addi %a28, %c1 : i32
    %a30 = arith.addi %a29, %c1 : i32
    %a31 = arith.addi %a30, %c1 : i32
    %a32 = arith.addi %a31, %c1 : i32
    memref.store %a32, %buf[%c0] : memref<1xi32>

    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %buf[%c0] : memref<1xi32>
    memref.dealloc %buf : memref<1xi32>
    return %v : i32
  }
}

// CHECK-LABEL: func.func @remat_long_chain
// CHECK-NOT:     memref.load
// CHECK:         return
