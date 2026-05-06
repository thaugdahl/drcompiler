// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// Two consecutive may-zero-trip affine.for loops, both bounded by the
// same SSA value that also sizes the alloc. The store loop and the
// load loop are correlated: when the trip count is zero neither runs.
// The store-loop join must drop the absent-path sentinel for the
// alloc, so loads inside the second loop classify SINGLE.

#id = affine_map<(d) -> (d)>

module {
  func.func @cross_loop(%arg0: memref<?x3xf32>) -> memref<?x3xf32> {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?x3xf32>
    %buf = memref.alloc(%dim) {alignment = 16 : i64} : memref<?x3xf32>
    %out = memref.alloc(%dim) {alignment = 16 : i64} : memref<?x3xf32>
    affine.for %i = 0 to #id(%dim) {
      affine.for %j = 0 to 3 {
        %v = affine.load %arg0[%i, %j] : memref<?x3xf32>
        affine.store %v, %buf[%i, %j] : memref<?x3xf32>
      }
    }
    affine.for %i = 0 to #id(%dim) {
      affine.for %j = 0 to 3 {
        // expected-remark @below {{load: SINGLE}}
        %v = affine.load %buf[%i, %j] : memref<?x3xf32>
        affine.store %v, %out[%i, %j] : memref<?x3xf32>
      }
    }
    return %out : memref<?x3xf32>
  }
}
