// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// Strategy D: chained inner loads.
//
// Reader's load of @g is SINGLE-provenance with the store in @writer.
// @writer's stored value depends on a load of @h. @h is itself written by
// @writer_h. The plan-builder rejects the @h load as a hoist (because
// writer_h transitively writes @h), then falls back to a recursive
// sub-plan that materializes @writer_h's stored value at the outer caller,
// anchored just-after the inner-writer-call.

module {
  // expected-remark @below {{cost-model:}}
  memref.global "private" @g : memref<i32> = uninitialized
  // expected-remark @below {{cost-model:}}
  memref.global "private" @h : memref<i32> = uninitialized

  func.func private @writer_h(%y: i32) {
    %h = memref.get_global @h : memref<i32>
    memref.store %y, %h[] : memref<i32>
    return
  }

  func.func private @writer(%x: i32) {
    %h = memref.get_global @h : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %hv = memref.load %h[] : memref<i32>
    %v = arith.addi %x, %hv : i32
    %g = memref.get_global @g : memref<i32>
    memref.store %v, %g[] : memref<i32>
    return
  }

  func.func private @reader() -> i32 {
    %g = memref.get_global @g : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func @run(%x: i32, %y: i32) -> i32 {
    call @writer_h(%y) : (i32) -> ()
    // (writer_h, writer) cross-group: @writer is the reader-of-h.
    // expected-remark @below {{interproc-cross: ACCEPT}}
    call @writer(%x) : (i32) -> ()
    // (writer, reader) cross-group, with chained sub-plan into writer_h.
    // expected-remark @below {{interproc-cross: ACCEPT}}
    %r = call @reader() : () -> i32
    return %r : i32
  }
}

// CHECK-LABEL: func.func @run
// CHECK:         call @writer_h
// After D: the inner h-value is materialized just-after writer_h, the
// outer addi is materialized just-after writer, and the reader takes the
// value as an extra arg.
// CHECK:         call @writer
// CHECK:         arith.addi
// CHECK:         call @reader
