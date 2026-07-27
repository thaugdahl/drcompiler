// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// A020: multiple stores exist for the alloc but all at disjoint indices.
// Load at index 0 has no covering store -> KILLED.

module {
  func.func @killed_disjoint_coverage() {
    %alloc = memref.alloc() : memref<32xi32>
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c0 = arith.constant 0 : index
    %idx2 = arith.constant 2 : index
    %idx3 = arith.constant 3 : index
    %idx4 = arith.constant 4 : index
    memref.store %c1, %alloc[%idx2] : memref<32xi32>
    memref.store %c2, %alloc[%idx3] : memref<32xi32>
    memref.store %c3, %alloc[%idx4] : memref<32xi32>
    // expected-remark @below {{load: KILLED}}
    %v = memref.load %alloc[%c0] : memref<32xi32>
    memref.dealloc %alloc : memref<32xi32>
    return
  }
}
