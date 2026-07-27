// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// A019: load at index 0, store only at index 1. Disjoint coverage ->
// the load appears in provenance (because the alloc has stores) but
// with an empty provenance set -> KILLED.

module {
  func.func @killed_no_store() {
    %alloc = memref.alloc() : memref<32xi32>
    %c42 = arith.constant 42 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    memref.store %c42, %alloc[%c1] : memref<32xi32>
    // expected-remark @below {{load: KILLED}}
    %v = memref.load %alloc[%c0] : memref<32xi32>
    memref.dealloc %alloc : memref<32xi32>
    return
  }
}
