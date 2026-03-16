// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// affine.store before call to external function: load after call is LEAKED.

module {
  func.func private @external_func(%arg: memref<32xi32>)

  func.func @test() {
    %alloc = memref.alloc() : memref<32xi32>
    %c42 = arith.constant 42 : i32
    affine.store %c42, %alloc[0] : memref<32xi32>
    func.call @external_func(%alloc) : (memref<32xi32>) -> ()
    // expected-remark @below {{load: LEAKED}}
    %val = affine.load %alloc[0] : memref<32xi32>
    memref.dealloc %alloc : memref<32xi32>
    return
  }
}
