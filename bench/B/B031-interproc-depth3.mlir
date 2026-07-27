// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// B031: Interproc depth-3 — three-hop chain.
// @wa stores to @a, @wb reads @a stores to @b, @wc reads @b stores to @c, @reader loads @c.

module {
  // expected-remark @+1 {{cost-model:}}
  memref.global "private" @a : memref<i32> = uninitialized
  // expected-remark @+1 {{cost-model:}}
  memref.global "private" @b : memref<i32> = uninitialized
  // expected-remark @+1 {{cost-model:}}
  memref.global "private" @c : memref<i32> = uninitialized

  func.func private @wa(%va: i32) {
    %a = memref.get_global @a : memref<i32>
    memref.store %va, %a[] : memref<i32>
    return
  }

  func.func private @wb() {
    %a = memref.get_global @a : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %av = memref.load %a[] : memref<i32>
    %c1 = arith.constant 1 : i32
    %vb = arith.addi %av, %c1 : i32
    %b = memref.get_global @b : memref<i32>
    memref.store %vb, %b[] : memref<i32>
    return
  }

  func.func private @wc() {
    %b = memref.get_global @b : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %bv = memref.load %b[] : memref<i32>
    %c2 = arith.constant 2 : i32
    %vc = arith.muli %bv, %c2 : i32
    %c = memref.get_global @c : memref<i32>
    memref.store %vc, %c[] : memref<i32>
    return
  }

  func.func private @reader() -> i32 {
    %c = memref.get_global @c : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %c[] : memref<i32>
    return %v : i32
  }

  func.func @run(%va: i32) -> i32 {
    call @wa(%va) : (i32) -> ()
    // expected-remark @below {{interproc-cross: ACCEPT_INPLACE}}
    call @wb() : () -> ()
    // expected-remark @below {{interproc-cross: ACCEPT_INPLACE}}
    call @wc() : () -> ()
    // expected-remark @below {{interproc-cross: ACCEPT_INPLACE}}
    %r = call @reader() : () -> i32
    return %r : i32
  }
}

// CHECK-LABEL: func.func @run
// CHECK:         call @wa
// CHECK:         call @wb
// CHECK:         call @wc
// CHECK:         call @reader
