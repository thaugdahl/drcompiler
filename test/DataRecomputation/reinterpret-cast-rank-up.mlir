// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// Phase B: offset-sensitive coverage through reinterpret_cast.
// Store happens at rank-2 indices `[0, 0]` to a memref<2x4xi32> backing
// memory. A rank-3 reinterpret_cast view of that memory is loaded at
// `[0, 0, 0]`. Both addresses linearize to base offset 0, so SINGLE
// classification follows a single overlapping store.

module {
  func.func @test() {
    %alloc = memref.alloc() : memref<2x4xi32>
    %c42 = arith.constant 42 : i32
    %c0 = arith.constant 0 : index

    memref.store %c42, %alloc[%c0, %c0] : memref<2x4xi32>

    %rc = memref.reinterpret_cast %alloc to offset: [0], sizes: [1, 2, 4],
        strides: [8, 4, 1]
        : memref<2x4xi32> to memref<1x2x4xi32, strided<[8, 4, 1], offset: 0>>

    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %rc[%c0, %c0, %c0] : memref<1x2x4xi32, strided<[8, 4, 1], offset: 0>>
    memref.dealloc %alloc : memref<2x4xi32>
    return
  }
}
