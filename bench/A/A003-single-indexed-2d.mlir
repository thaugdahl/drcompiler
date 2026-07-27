// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// A003: 2D memref, nested affine store + load (64x64).

module {
  func.func @single_indexed_2d() {
    %alloc = memref.alloc() : memref<64x64xf32>
    %c = arith.constant 1.0 : f32
    affine.for %i = 0 to 64 {
      affine.for %j = 0 to 64 {
        affine.store %c, %alloc[%i, %j] : memref<64x64xf32>
      }
    }
    affine.for %i = 0 to 64 {
      affine.for %j = 0 to 64 {
        // expected-remark @below {{load: SINGLE}}
        %v = affine.load %alloc[%i, %j] : memref<64x64xf32>
      }
    }
    memref.dealloc %alloc : memref<64x64xf32>
    return
  }
}
