// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// Strategy D: depth-2 chain.
//
// @writer_a stores a, @writer_b stores b that depends on @a, @writer_c
// stores c that depends on @b, @reader loads @c. All four functions
// participate in a chained materialization at the outer caller.

module {
  // expected-remark @below {{cost-model:}}
  memref.global "private" @a : memref<i32> = uninitialized
  // expected-remark @below {{cost-model:}}
  memref.global "private" @b : memref<i32> = uninitialized
  // expected-remark @below {{cost-model:}}
  memref.global "private" @c : memref<i32> = uninitialized

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

  func.func private @writer_c() {
    %b = memref.get_global @b : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %bv = memref.load %b[] : memref<i32>
    %two = arith.constant 2 : i32
    %vc = arith.muli %bv, %two : i32
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
    call @writer_a(%va) : (i32) -> ()
    // (writer_a, writer_b) — @writer_b is reader of @a.
    // expected-remark @below {{interproc-cross: ACCEPT}}
    call @writer_b() : () -> ()
    // (writer_b, writer_c) — @writer_c is reader of @b.
    // expected-remark @below {{interproc-cross: ACCEPT}}
    call @writer_c() : () -> ()
    // (writer_c, @reader) — chained sub-plan reaches all the way back.
    // expected-remark @below {{interproc-cross: ACCEPT}}
    %r = call @reader() : () -> i32
    return %r : i32
  }
}

// CHECK-LABEL: func.func @run
// All three writers' calls remain (their args were threaded), and the
// reader receives the chained materialization.
// CHECK:         call @writer_a
// CHECK:         call @writer_b
// CHECK:         call @writer_c
// CHECK:         call @reader
