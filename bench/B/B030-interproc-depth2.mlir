// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// B030: Interproc depth-2 — store in callee's callee, load in top caller.
// @writer_a stores to @a, @writer_b reads @a and stores to @b, @reader loads @b.

module {
  // expected-remark @+1 {{cost-model:}}
  memref.global "private" @a : memref<i32> = uninitialized
  // expected-remark @+1 {{cost-model:}}
  memref.global "private" @b : memref<i32> = uninitialized

  func.func private @writer_a(%va: i32) {
    %a = memref.get_global @a : memref<i32>
    memref.store %va, %a[] : memref<i32>
    return
  }

  func.func private @writer_b() {
    %a = memref.get_global @a : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %av = memref.load %a[] : memref<i32>
    %one = arith.constant 1 : i32
    %vb = arith.addi %av, %one : i32
    %b = memref.get_global @b : memref<i32>
    memref.store %vb, %b[] : memref<i32>
    return
  }

  func.func private @reader() -> i32 {
    %b = memref.get_global @b : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %b[] : memref<i32>
    return %v : i32
  }

  func.func @run(%va: i32) -> i32 {
    call @writer_a(%va) : (i32) -> ()
    // expected-remark @below {{interproc-cross: ACCEPT_INPLACE}}
    call @writer_b() : () -> ()
    // expected-remark @below {{interproc-cross: ACCEPT_INPLACE}}
    %r = call @reader() : () -> i32
    return %r : i32
  }
}

// CHECK-LABEL: func.func @run
// CHECK:         call @writer_a
// CHECK:         call @writer_b
// CHECK:         call @reader
