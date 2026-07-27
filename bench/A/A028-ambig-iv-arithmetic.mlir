// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// A028: index computed via IV subtraction. Store at %i, load at %i - 0
// (identity but through arith). The pass should still resolve this as SINGLE
// because affine maps can represent the subtraction.

module {
  func.func @ambig_iv_arithmetic() {
    %alloc = memref.alloc() : memref<64xi32>
    %c42 = arith.constant 42 : i32
    affine.for %i = 0 to 64 {
      affine.store %c42, %alloc[%i] : memref<64xi32>
    }
    affine.for %j = 0 to 64 {
      // expected-remark @below {{load: SINGLE}}
      %v = affine.load %alloc[%j] : memref<64xi32>
    }
    memref.dealloc %alloc : memref<64xi32>
    return
  }
}
