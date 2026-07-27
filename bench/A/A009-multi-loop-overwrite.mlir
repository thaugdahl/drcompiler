// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// A009: store in scf.for loop body, pre-loop store, load after loop.
// The loop store creates a second provenance -> MULTI.

module {
  func.func @multi_loop_overwrite() {
    %alloc = memref.alloc() : memref<i32>
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1_idx = arith.constant 1 : index
    %val1 = arith.constant 1 : i32
    %val2 = arith.constant 2 : i32
    memref.store %val1, %alloc[] : memref<i32>
    scf.for %i = %c0 to %c10 step %c1_idx {
      memref.store %val2, %alloc[] : memref<i32>
    }
    // expected-remark @below {{load: MULTI}}
    %v = memref.load %alloc[] : memref<i32>
    memref.dealloc %alloc : memref<i32>
    return
  }
}
