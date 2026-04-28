// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// Strategy 4: cross-function ordered recomputation.
//
// Caller @run calls @writer (which stores x+1 to @g), then @reader (which
// loads @g). Strategy 4 should:
//   1. Recognize the load in @reader is SINGLE-provenance with the store
//      in @writer.
//   2. Find the conforming caller @run where @writer's call dominates
//      @reader's call with no intervening writer to @g.
//   3. Specialize @reader (or rewrite in place since it's private with one
//      caller) to take the value as an extra arg, eliminating the load.

module {
  // expected-remark @below {{cost-model:}}
  memref.global "private" @g : memref<i32> = uninitialized

  func.func private @writer(%x: i32) {
    %g = memref.get_global @g : memref<i32>
    %one = arith.constant 1 : i32
    %v = arith.addi %x, %one : i32
    memref.store %v, %g[] : memref<i32>
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
    // expected-remark @below {{interproc-cross: ACCEPT}}
    %r = call @reader() : () -> i32
    return %r : i32
  }
}

// After Strategy 4: @reader's load is replaced by an arg threaded from @run.
// CHECK-LABEL: func.func @run
// CHECK:         call @writer
// CHECK:         %[[ONE:.*]] = arith.constant 1
// CHECK:         %[[V:.*]] = arith.addi %{{.*}}, %[[ONE]]
// CHECK:         %[[R:.*]] = call @reader(%[[V]]
// CHECK:         return %[[R]]
