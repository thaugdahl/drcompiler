// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// A021: rank-0 store to a different buffer, indexed store to alloc at
// index 5. Load at index 0 has no covering store -> KILLED.
// The rank-0 store to a separate alloc ensures StoreMap entries exist but
// are irrelevant for the indexed alloc's load.

module {
  func.func @killed_rank0_indexed() {
    %alloc = memref.alloc() : memref<16xi32>
    %scalar = memref.alloc() : memref<i32>
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index

    // Store to indexed buffer at index 5 only.
    memref.store %c1, %alloc[%c5] : memref<16xi32>
    // Store to separate scalar buffer.
    memref.store %c2, %scalar[] : memref<i32>

    // Load at index 0: no store covers this index.
    // expected-remark @below {{load: KILLED}}
    %v = memref.load %alloc[%c0] : memref<16xi32>
    // expected-remark @below {{load: SINGLE}}
    %s = memref.load %scalar[] : memref<i32>

    memref.dealloc %alloc : memref<16xi32>
    memref.dealloc %scalar : memref<i32>
    return
  }
}
