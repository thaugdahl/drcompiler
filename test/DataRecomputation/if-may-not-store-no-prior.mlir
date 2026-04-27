// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// S1-A: One-sided scf.if with no prior store. The else-path has no reaching
// store, so the post-join load must NOT be SINGLE.

module {
  func.func @if_may_not_store(%cond: i1) -> i32 {
    %alloc = memref.alloc() : memref<i32>
    %c42 = arith.constant 42 : i32
    scf.if %cond {
      memref.store %c42, %alloc[] : memref<i32>
    }
    // expected-remark @below {{load: LEAKED}}
    %v = memref.load %alloc[] : memref<i32>
    memref.dealloc %alloc : memref<i32>
    return %v : i32
  }
}
