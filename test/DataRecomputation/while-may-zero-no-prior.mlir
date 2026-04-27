// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// S1-A: scf.while where condition may be false on first iteration, with no
// prior store. Post-loop load must NOT be SINGLE.

module {
  func.func @while_may_zero(%cond: i1) -> i32 {
    %alloc = memref.alloc() : memref<i32>
    %c42 = arith.constant 42 : i32
    scf.while : () -> () {
      scf.condition(%cond)
    } do {
      memref.store %c42, %alloc[] : memref<i32>
      scf.yield
    }
    // expected-remark @below {{load: LEAKED}}
    %v = memref.load %alloc[] : memref<i32>
    memref.dealloc %alloc : memref<i32>
    return %v : i32
  }
}
