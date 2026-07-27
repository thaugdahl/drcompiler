// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// A018: mixed provenance within same buffer. The external call receives a
// subview of the upper half, but the pass conservatively marks all loads
// from the underlying alloc as LEAKED (the clobber is whole-allocation).

module {
  func.func private @external_func(%m: memref<16xi32, strided<[1], offset: 16>>)

  func.func @leaked_partial() {
    %alloc = memref.alloc() : memref<32xi32>
    %c42 = arith.constant 42 : i32
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index

    // Store at index 0.
    memref.store %c42, %alloc[%c0] : memref<32xi32>

    // Pass upper half to external function.
    %sv = memref.subview %alloc[16][16][1]
        : memref<32xi32> to memref<16xi32, strided<[1], offset: 16>>
    call @external_func(%sv) : (memref<16xi32, strided<[1], offset: 16>>) -> ()

    // Both loads are LEAKED because the pass conservatively clobbers the
    // entire allocation when any view escapes.
    // expected-remark @below {{load: LEAKED}}
    %v0 = memref.load %alloc[%c0] : memref<32xi32>
    // expected-remark @below {{load: LEAKED}}
    %v16 = memref.load %alloc[%c16] : memref<32xi32>

    memref.dealloc %alloc : memref<32xi32>
    return
  }
}
