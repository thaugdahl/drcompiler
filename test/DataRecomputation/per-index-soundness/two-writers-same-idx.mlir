// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// Two unconditional affine.stores at the same constant index — second
// kills the first (last-write-wins). Load must classify SINGLE on the
// surviving store. This documents the analyzer's existing semantics
// applied to the new affine path.

module {
  func.func @two_writers_same_const(%a: i64, %b: i64) -> i64 {
    %buf = memref.alloc() : memref<3xi64>
    %c0 = arith.constant 0 : index
    affine.store %a, %buf[%c0] : memref<3xi64>
    affine.store %b, %buf[%c0] : memref<3xi64>
    // expected-remark @below {{load: SINGLE}}
    %r = affine.load %buf[%c0] : memref<3xi64>
    return %r : i64
  }
}
