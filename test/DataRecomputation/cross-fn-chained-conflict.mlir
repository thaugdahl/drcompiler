// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// Strategy D — mutual conflict rejection.
//
// Same chain shape as the simple D test, but @writer_h is called BETWEEN
// the outer-writer-call and the reader-call in @run. The chained sub-plan
// must reject this caller because:
//   - the outer (writer, reader) plan can't be hoisted (call to writer_h
//     intervenes), and
//   - the sub-plan's inner-writer-call (writer_h) does not properly
//     dominate the outer-writer-call.
// Strategy 4 should NOT fire on the @reader call.

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
    // expected-remark @below {{interproc-cross: REJECT_NO_ORDERED_CALLER}}
    %hv = memref.load %h[] : memref<i32>
    %v = arith.addi %x, %hv : i32
    %g = memref.get_global @g : memref<i32>
    memref.store %v, %g[] : memref<i32>
    return
  }

  func.func private @reader() -> i32 {
    %g = memref.get_global @g : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    // expected-remark @below {{interproc-cross: REJECT_NO_ORDERED_CALLER}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func @run(%x: i32, %z: i32) -> i32 {
    // No call to @writer_h dominates @writer here, so the
    // (writer, reader) chained sub-plan rejects @run as a caller.
    call @writer(%x) : (i32) -> ()
    // Intervening call to @writer_h after @writer rules out the
    // sub-plan's "innerWC properly dominates outerWC" requirement.
    call @writer_h(%z) : (i32) -> ()
    %r = call @reader() : () -> i32
    return %r : i32
  }
}

// CHECK-LABEL: func.func @run
// The reader call survives unchanged (not specialized).
// CHECK:         call @writer
// CHECK:         call @writer_h
// CHECK:         %{{.*}} = call @reader()
