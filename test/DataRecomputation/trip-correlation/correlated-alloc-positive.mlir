// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// Alloc's dynamic dim is bound to the same SSA value driving the
// store-loop's upper bound. When the loop trips zero times, the alloc
// has zero elements along that dim, so any later in-bounds load is
// unreachable on that path. The absent-path sentinel must be skipped:
// the post-loop load classifies SINGLE rather than LEAKED.

#id = affine_map<(d) -> (d)>

module {
  func.func @correlated(%arg0: memref<?x4xf32>) -> f32 {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?x4xf32>
    %buf = memref.alloc(%dim) {alignment = 16 : i64} : memref<?x4xf32>
    affine.for %i = 0 to #id(%dim) {
      affine.for %j = 0 to 4 {
        %v = affine.load %arg0[%i, %j] : memref<?x4xf32>
        affine.store %v, %buf[%i, %j] : memref<?x4xf32>
      }
    }
    // expected-remark @below {{load: SINGLE}}
    %r = affine.load %buf[%c0, %c0] : memref<?x4xf32>
    return %r : f32
  }
}
