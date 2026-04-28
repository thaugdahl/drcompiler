// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// Positive: shape-tuple pattern. Three affine.stores inside trip-1
// affine.for loops with affine.apply shifts target distinct constant
// indices of a 3-element buffer. Affine.load at constant index 1 must
// see only the writer at idx 1.

#shift0 = affine_map<(d0) -> (d0)>
#shift1 = affine_map<(d0) -> (d0 + 1)>
#shift2 = affine_map<(d0) -> (d0 + 2)>

module {
  func.func @shape_tuple(%a: i64, %b: i64, %c: i64) -> i64 {
    %buf = memref.alloc() : memref<3xi64>
    affine.for %i = 0 to 1 {
      %ii = affine.apply #shift0(%i)
      affine.store %a, %buf[%ii] : memref<3xi64>
    }
    affine.for %i = 0 to 1 {
      %ii = affine.apply #shift1(%i)
      affine.store %b, %buf[%ii] : memref<3xi64>
    }
    affine.for %i = 0 to 1 {
      %ii = affine.apply #shift2(%i)
      affine.store %c, %buf[%ii] : memref<3xi64>
    }
    %c1 = arith.constant 1 : index
    // expected-remark @below {{load: SINGLE}}
    %r = affine.load %buf[%c1] : memref<3xi64>
    return %r : i64
  }
}
