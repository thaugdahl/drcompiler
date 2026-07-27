// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// F014: 3-deep nested affine.for (32x32x32) with store, followed by load.

module {
  func.func @nest_3_deep(%arg0: i32) -> i32 {
    %buf = memref.alloc() : memref<32x32x32xi32>
    %c1 = arith.constant 1 : i32
    affine.for %i = 0 to 32 {
      affine.for %j = 0 to 32 {
        affine.for %k = 0 to 32 {
          %ic = arith.index_cast %i : index to i32
          %jc = arith.index_cast %j : index to i32
          %kc = arith.index_cast %k : index to i32
          %v1 = arith.addi %ic, %jc : i32
          %v2 = arith.addi %v1, %kc : i32
          %w = arith.addi %v2, %arg0 : i32
          affine.store %w, %buf[%i, %j, %k] : memref<32x32x32xi32>
        }
      }
    }
    // expected-remark @below {{load: SINGLE}}
    %out = affine.load %buf[0, 0, 0] : memref<32x32x32xi32>
    memref.dealloc %buf : memref<32x32x32xi32>
    return %out : i32
  }
}
