// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// B015: Full remat — chain spans caller->callee via global.
// Callee computes arg+10 and stores to global; caller loads.

module {
  // expected-remark @below {{cost-model:}}
  memref.global "private" @g : memref<i32> = uninitialized

  func.func private @compute_and_store(%x: i32) {
    %c10 = arith.constant 10 : i32
    %val = arith.addi %x, %c10 : i32
    %g = memref.get_global @g : memref<i32>
    memref.store %val, %g[] : memref<i32>
    return
  }

  func.func private @reader() -> i32 {
    %g = memref.get_global @g : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func @run(%x: i32) -> i32 {
    call @compute_and_store(%x) : (i32) -> ()
    // expected-remark @below {{interproc-cross: ACCEPT_INPLACE}}
    %r = call @reader() : () -> i32
    return %r : i32
  }
}

// CHECK-LABEL: func.func @run
// CHECK:         call @compute_and_store
// CHECK:         call @reader
