// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// B029: Interproc — callee stores to two globals.

module {
  // expected-remark @below {{cost-model:}}
  memref.global "private" @g1 : memref<i32> = uninitialized
  // expected-remark @below {{cost-model:}}
  memref.global "private" @g2 : memref<i32> = uninitialized

  func.func private @writer(%x: i32, %y: i32) {
    %g1 = memref.get_global @g1 : memref<i32>
    %g2 = memref.get_global @g2 : memref<i32>
    memref.store %x, %g1[] : memref<i32>
    memref.store %y, %g2[] : memref<i32>
    return
  }

  func.func private @reader1() -> i32 {
    %g1 = memref.get_global @g1 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g1[] : memref<i32>
    return %v : i32
  }

  func.func private @reader2() -> i32 {
    %g2 = memref.get_global @g2 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g2[] : memref<i32>
    return %v : i32
  }

  func.func @run(%x: i32, %y: i32) -> i32 {
    call @writer(%x, %y) : (i32, i32) -> ()
    // expected-remark @below {{interproc-cross: ACCEPT_INPLACE}}
    %a = call @reader1() : () -> i32
    // expected-remark @below {{interproc-cross: ACCEPT_INPLACE}}
    %b = call @reader2() : () -> i32
    %r = arith.addi %a, %b : i32
    return %r : i32
  }
}

// CHECK-LABEL: func.func @run
// CHECK:         call @writer
// CHECK:         call @reader1
// CHECK:         call @reader2
