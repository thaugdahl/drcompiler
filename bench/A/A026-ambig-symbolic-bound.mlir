// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// A026: affine.for with symbolic upper bound. Store at each index in
// [0, %n), load at index 0. Symbolic bound means zero-trip possible,
// so provenance is LEAKED (may not have been written).

module {
  func.func @ambig_symbolic_bound(%n: index) {
    %alloc = memref.alloc() : memref<256xi32>
    %c42 = arith.constant 42 : i32
    affine.for %i = 0 to %n {
      affine.store %c42, %alloc[%i] : memref<256xi32>
    }
    // expected-remark @below {{load: LEAKED}}
    %v = affine.load %alloc[0] : memref<256xi32>
    memref.dealloc %alloc : memref<256xi32>
    return
  }
}
