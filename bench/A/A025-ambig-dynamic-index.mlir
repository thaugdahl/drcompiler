// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// A025: dynamic (non-affine) store index. The index is a function argument
// so the pass cannot determine coverage -> conservative classification.

module {
  func.func @ambig_dynamic_index(%idx: index) {
    %alloc = memref.alloc() : memref<32xi32>
    %c42 = arith.constant 42 : i32
    %c0 = arith.constant 0 : index
    memref.store %c42, %alloc[%idx] : memref<32xi32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %alloc[%c0] : memref<32xi32>
    memref.dealloc %alloc : memref<32xi32>
    return
  }
}
