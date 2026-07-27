// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// F015: 4-deep nested affine.for (16x16x16x16) with store, followed by load.

module {
  func.func @nest_4_deep(%arg0: i32) -> i32 {
    %buf = memref.alloc() : memref<16x16x16x16xi32>
    affine.for %i = 0 to 16 {
      affine.for %j = 0 to 16 {
        affine.for %k = 0 to 16 {
          affine.for %l = 0 to 16 {
            %ic = arith.index_cast %i : index to i32
            %jc = arith.index_cast %j : index to i32
            %kc = arith.index_cast %k : index to i32
            %lc = arith.index_cast %l : index to i32
            %v1 = arith.addi %ic, %jc : i32
            %v2 = arith.addi %v1, %kc : i32
            %v3 = arith.addi %v2, %lc : i32
            %w = arith.addi %v3, %arg0 : i32
            affine.store %w, %buf[%i, %j, %k, %l] : memref<16x16x16x16xi32>
          }
        }
      }
    }
    // expected-remark @below {{load: SINGLE}}
    %out = affine.load %buf[0, 0, 0, 0] : memref<16x16x16x16xi32>
    memref.dealloc %buf : memref<16x16x16x16xi32>
    return %out : i32
  }
}
