// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// A symbolic-index writer (idx not foldable) coexists with a const-index
// writer at idx 0. Symbolic may overlap idx 0, so const load at 0 must
// NOT classify SINGLE.

module {
  func.func @symbolic_poisons(%i: index, %v: i64, %w: i64) -> i64 {
    %buf = memref.alloc() : memref<3xi64>
    %c0 = arith.constant 0 : index
    affine.store %v, %buf[%c0] : memref<3xi64>
    memref.store %w, %buf[%i] : memref<3xi64>
    // expected-remark @below {{load: MULTI}}
    %r = affine.load %buf[%c0] : memref<3xi64>
    return %r : i64
  }
}
