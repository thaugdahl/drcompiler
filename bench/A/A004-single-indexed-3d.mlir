// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// A004: 3D memref, triple-nested affine store + load (16x16x16).

module {
  func.func @single_indexed_3d() {
    %alloc = memref.alloc() : memref<16x16x16xf32>
    %c = arith.constant 1.0 : f32
    affine.for %i = 0 to 16 {
      affine.for %j = 0 to 16 {
        affine.for %k = 0 to 16 {
          affine.store %c, %alloc[%i, %j, %k] : memref<16x16x16xf32>
        }
      }
    }
    affine.for %i = 0 to 16 {
      affine.for %j = 0 to 16 {
        affine.for %k = 0 to 16 {
          // expected-remark @below {{load: SINGLE}}
          %v = affine.load %alloc[%i, %j, %k] : memref<16x16x16xf32>
        }
      }
    }
    memref.dealloc %alloc : memref<16x16x16xf32>
    return
  }
}
