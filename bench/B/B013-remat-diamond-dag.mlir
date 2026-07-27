// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// B013: Full remat — DAG computation (two ops share one operand).
//   x -> a = x+1
//        b = x+2
//   c = a + b  (diamond: both a and b depend on x)

module {
  func.func @remat_diamond_dag(%x: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32

    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %buf = memref.alloc() : memref<1xi32>

    %a = arith.addi %x, %c1 : i32
    %b = arith.addi %x, %c2 : i32
    %c = arith.addi %a, %b : i32
    memref.store %c, %buf[%c0] : memref<1xi32>

    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %buf[%c0] : memref<1xi32>
    memref.dealloc %buf : memref<1xi32>
    return %v : i32
  }
}

// CHECK-LABEL: func.func @remat_diamond_dag
// CHECK-NOT:     memref.load
// CHECK:         return
