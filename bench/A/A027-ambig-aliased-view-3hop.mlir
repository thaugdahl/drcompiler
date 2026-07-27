// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// A027: 3-hop view chain: alloc -> subview -> reinterpret_cast -> subview.
// Store through the 3-hop chain at logical [0] maps to alloc[8].
// Load from alloc[8] should resolve as SINGLE.

module {
  func.func @ambig_aliased_view_3hop() {
    %alloc = memref.alloc() : memref<32xi32>
    %c42 = arith.constant 42 : i32
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index

    // Hop 1: subview starting at offset 4.
    %sv1 = memref.subview %alloc[4][16][1]
        : memref<32xi32> to memref<16xi32, strided<[1], offset: 4>>

    // Hop 2: reinterpret_cast (identity, preserves offset).
    %rc = memref.reinterpret_cast %sv1 to offset: [4], sizes: [16], strides: [1]
        : memref<16xi32, strided<[1], offset: 4>> to memref<16xi32, strided<[1], offset: 4>>

    // Hop 3: subview of the reinterpret_cast, starting at local offset 4
    // (which is base offset 4+4 = 8).
    %sv2 = memref.subview %rc[4][8][1]
        : memref<16xi32, strided<[1], offset: 4>> to memref<8xi32, strided<[1], offset: 8>>

    memref.store %c42, %sv2[%c0] : memref<8xi32, strided<[1], offset: 8>>

    // Load from alloc[8] — should match the store through the 3-hop chain.
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %alloc[%c8] : memref<32xi32>
    memref.dealloc %alloc : memref<32xi32>
    return
  }
}
