// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// Mixed: memref.alloc with memref.store, then affine.load.
// Both dialects should be tracked and cross-reference correctly.

module {
  func.func @test() {
    %alloc = memref.alloc() : memref<32xi32>
    %c42 = arith.constant 42 : i32
    %idx = arith.constant 0 : index
    memref.store %c42, %alloc[%idx] : memref<32xi32>
    // expected-remark @below {{load: SINGLE}}
    %val = affine.load %alloc[0] : memref<32xi32>
    memref.dealloc %alloc : memref<32xi32>
    return
  }
}
