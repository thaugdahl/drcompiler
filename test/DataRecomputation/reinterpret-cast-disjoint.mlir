// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// Phase B: offset-sensitive coverage rules out a non-aliasing store at a
// disjoint linear offset. Two stores via different reinterpret_casts hit
// linear positions 0 and 4 of the same alloc; the load reads position 0.
// With offset-sensitive analysis the load's provenance is SINGLE — only
// the store at position 0 reaches it. Pre-Phase B this would have been
// MULTI (any store through a view degraded coverage to may-write-anywhere).

module {
  func.func @test() {
    %alloc = memref.alloc() : memref<8xi32>
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c0 = arith.constant 0 : index

    %rc_lo = memref.reinterpret_cast %alloc to offset: [0], sizes: [4], strides: [1]
        : memref<8xi32> to memref<4xi32, strided<[1], offset: 0>>
    %rc_hi = memref.reinterpret_cast %alloc to offset: [4], sizes: [4], strides: [1]
        : memref<8xi32> to memref<4xi32, strided<[1], offset: 4>>

    memref.store %c1, %rc_lo[%c0] : memref<4xi32, strided<[1], offset: 0>>
    memref.store %c2, %rc_hi[%c0] : memref<4xi32, strided<[1], offset: 4>>

    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %alloc[%c0] : memref<8xi32>
    memref.dealloc %alloc : memref<8xi32>
    return
  }
}
