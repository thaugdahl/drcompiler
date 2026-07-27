// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// A015: memref has its pointer extracted via extract_aligned_pointer_as_index.
// Despite the pointer escape, the store provenance is unaffected because no
// external write occurs through the pointer. Classification remains SINGLE.
// (The escape analysis would flag this for buffer-elimination gating, but
// provenance classification is purely store-based.)

module {
  func.func @leaked_ptr_cast() -> index {
    %alloc = memref.alloc() : memref<32xi32>
    %c42 = arith.constant 42 : i32
    %c0 = arith.constant 0 : index
    memref.store %c42, %alloc[%c0] : memref<32xi32>
    %ptr = memref.extract_aligned_pointer_as_index %alloc : memref<32xi32> -> index
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %alloc[%c0] : memref<32xi32>
    memref.dealloc %alloc : memref<32xi32>
    return %ptr : index
  }
}
