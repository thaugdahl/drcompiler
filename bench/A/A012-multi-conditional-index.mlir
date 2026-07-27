// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// A012: store at i or i+1 depending on condition.
// Load at index 0 may be reached by either store -> MULTI.

module {
  func.func @multi_conditional_index(%cond: i1) {
    %alloc = memref.alloc() : memref<32xi32>
    %c42 = arith.constant 42 : i32
    %c99 = arith.constant 99 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.if %cond {
      memref.store %c42, %alloc[%c0] : memref<32xi32>
    } else {
      memref.store %c99, %alloc[%c0] : memref<32xi32>
    }
    // expected-remark @below {{load: MULTI}}
    %v = memref.load %alloc[%c0] : memref<32xi32>
    memref.dealloc %alloc : memref<32xi32>
    return
  }
}
