// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// Phase B: subview composing with the alloc's identity layout linearizes
// correctly. The subview targets `%alloc[16][16][1]` — flat positions
// 16..31 of a 32-element buffer. A store at subview index 0 lands at
// linear 16; a load of `%alloc[16]` linearizes to the same offset.

module {
  func.func @test() {
    %alloc = memref.alloc() : memref<32xi32>
    %c42 = arith.constant 42 : i32
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index

    %sv = memref.subview %alloc[16][16][1]
        : memref<32xi32> to memref<16xi32, strided<[1], offset: 16>>

    memref.store %c42, %sv[%c0] : memref<16xi32, strided<[1], offset: 16>>

    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %alloc[%c16] : memref<32xi32>
    memref.dealloc %alloc : memref<32xi32>
    return
  }
}
