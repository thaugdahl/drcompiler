// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// A002: 1D memref, affine store + load at same index range (trip 256).

module {
  func.func @single_indexed_1d() {
    %alloc = memref.alloc() : memref<256xf32>
    %c = arith.constant 1.0 : f32
    affine.for %i = 0 to 256 {
      affine.store %c, %alloc[%i] : memref<256xf32>
    }
    affine.for %j = 0 to 256 {
      // expected-remark @below {{load: SINGLE}}
      %v = affine.load %alloc[%j] : memref<256xf32>
    }
    memref.dealloc %alloc : memref<256xf32>
    return
  }
}
