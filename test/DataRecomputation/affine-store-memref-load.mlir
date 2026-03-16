// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// affine.store followed by memref.load: cross-dialect provenance.

module {
  func.func @test() {
    %alloc = memref.alloc() : memref<32xi32>
    %c42 = arith.constant 42 : i32
    %idx = arith.constant 0 : index
    affine.store %c42, %alloc[%idx] : memref<32xi32>
    // expected-remark @below {{load: SINGLE}}
    %val = memref.load %alloc[%idx] : memref<32xi32>
    memref.dealloc %alloc : memref<32xi32>
    return
  }
}
