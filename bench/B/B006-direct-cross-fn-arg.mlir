// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// B006: Direct forward — callee stores caller's arg via global, caller loads.

module {
  // expected-remark @below {{cost-model:}}
  memref.global "private" @g : memref<i32> = uninitialized

  func.func private @writer(%x: i32) {
    %g = memref.get_global @g : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @reader() -> i32 {
    %g = memref.get_global @g : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func @run(%x: i32) -> i32 {
    call @writer(%x) : (i32) -> ()
    // expected-remark @below {{interproc-cross: ACCEPT_INPLACE}}
    %r = call @reader() : () -> i32
    return %r : i32
  }
}

// CHECK-LABEL: func.func @run
// CHECK:         call @writer
// CHECK:         call @reader
