// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// A024: store appears AFTER load in execution order.
// A pre-existing store at index 5 puts the alloc into StoreMap,
// but no store covers index 0 before the load -> KILLED.
// The post-load store at index 0 does not reach the load.

module {
  func.func @killed_store_after_load() {
    %alloc = memref.alloc() : memref<32xi32>
    %c42 = arith.constant 42 : i32
    %c99 = arith.constant 99 : i32
    %c0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index
    memref.store %c42, %alloc[%c5] : memref<32xi32>
    // expected-remark @below {{load: KILLED}}
    %v = memref.load %alloc[%c0] : memref<32xi32>
    memref.store %c99, %alloc[%c0] : memref<32xi32>
    memref.dealloc %alloc : memref<32xi32>
    return
  }
}
