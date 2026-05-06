// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// Phase B: a 2-hop view chain (subview then reinterpret_cast) composes
// into the alloc's flat layout. Store at the chained view's `[0]`
// resolves to alloc linear offset 4 (the subview's offset). The load
// directly addresses alloc[4] at rank 1 — same linear offset → SINGLE.

module {
  func.func @test() {
    %alloc = memref.alloc() : memref<16xi32>
    %c42 = arith.constant 42 : i32
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index

    %sv = memref.subview %alloc[4][8][1]
        : memref<16xi32> to memref<8xi32, strided<[1], offset: 4>>
    %rc = memref.reinterpret_cast %sv to offset: [4], sizes: [8], strides: [1]
        : memref<8xi32, strided<[1], offset: 4>> to memref<8xi32, strided<[1], offset: 4>>

    memref.store %c42, %rc[%c0] : memref<8xi32, strided<[1], offset: 4>>

    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %alloc[%c4] : memref<16xi32>
    memref.dealloc %alloc : memref<16xi32>
    return
  }
}
