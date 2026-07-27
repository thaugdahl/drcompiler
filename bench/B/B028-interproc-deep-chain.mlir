// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// B028: Interproc — callee stores a computed chain result to global.
// Uses arg + constant chain that the pass can rematerialize.

module {
  // expected-remark @+1 {{cost-model:}}
  memref.global "private" @g : memref<i32> = uninitialized

  func.func private @compute(%x: i32) {
    %c10 = arith.constant 10 : i32
    %c3 = arith.constant 3 : i32
    %a = arith.addi %x, %c10 : i32
    %b = arith.muli %a, %c3 : i32
    %g = memref.get_global @g : memref<i32>
    memref.store %b, %g[] : memref<i32>
    return
  }

  func.func private @reader() -> i32 {
    %g = memref.get_global @g : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func @run(%x: i32) -> i32 {
    call @compute(%x) : (i32) -> ()
    // expected-remark @below {{interproc-cross: ACCEPT_INPLACE}}
    %r = call @reader() : () -> i32
    return %r : i32
  }
}

// CHECK-LABEL: func.func @run
// CHECK:         call @compute
// Rematerialized chain at caller.
// CHECK:         arith.addi
// CHECK:         arith.muli
// CHECK:         call @reader
