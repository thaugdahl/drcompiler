// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// A023: store only in if-branch (no else), different index from load.
// Conditional store introduces uncertainty → LEAKED provenance.

module {
  func.func @killed_conditional_store(%cond: i1) {
    %alloc = memref.alloc() : memref<32xi32>
    %c42 = arith.constant 42 : i32
    %c0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index
    scf.if %cond {
      memref.store %c42, %alloc[%c5] : memref<32xi32>
    }
    // expected-remark @below {{load: LEAKED}}
    %v = memref.load %alloc[%c0] : memref<32xi32>
    memref.dealloc %alloc : memref<32xi32>
    return
  }
}
