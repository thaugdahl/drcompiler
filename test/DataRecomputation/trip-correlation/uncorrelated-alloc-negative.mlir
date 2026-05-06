// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// Negative case for the trip-correlation rule: alloc's dynamic dim
// (%m) is independent of the loop's upper bound (%n). When %n == 0
// the loop is zero-trip, but the alloc may still hold valid elements
// along the m-dim and a later load can observe an uninitialised value.
// The absent-path sentinel must be preserved -> LEAKED.

module {
  func.func @uncorrelated(%n: index, %m: index, %v: f32) -> f32 {
    %c0 = arith.constant 0 : index
    %buf = memref.alloc(%m) {alignment = 16 : i64} : memref<?xf32>
    affine.for %i = 0 to %n {
      memref.store %v, %buf[%c0] : memref<?xf32>
    }
    // expected-remark @below {{load: LEAKED}}
    %r = memref.load %buf[%c0] : memref<?xf32>
    return %r : f32
  }
}
