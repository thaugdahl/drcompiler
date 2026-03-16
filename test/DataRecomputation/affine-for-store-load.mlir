// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// affine.for with affine.store in the body, affine.load after: SINGLE
// because affine.for with static lb=0 < ub=10 always executes.

module {
  func.func @test() {
    %alloc = memref.alloc() : memref<32xi32>
    %c42 = arith.constant 42 : i32
    affine.for %i = 0 to 10 {
      affine.store %c42, %alloc[%i] : memref<32xi32>
    }
    // expected-remark @below {{load: SINGLE}}
    %val = affine.load %alloc[0] : memref<32xi32>
    memref.dealloc %alloc : memref<32xi32>
    return
  }
}
