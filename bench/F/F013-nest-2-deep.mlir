// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// F013: 2-deep nested affine.for (64x64) with store, followed by load.

module {
  func.func @nest_2_deep(%arg0: i32) -> i32 {
    %buf = memref.alloc() : memref<64x64xi32>
    %c1 = arith.constant 1 : i32
    affine.for %i = 0 to 64 {
      affine.for %j = 0 to 64 {
        %ic = arith.index_cast %i : index to i32
        %jc = arith.index_cast %j : index to i32
        %v = arith.addi %ic, %jc : i32
        %w = arith.addi %v, %arg0 : i32
        affine.store %w, %buf[%i, %j] : memref<64x64xi32>
      }
    }
    // expected-remark @below {{load: SINGLE}}
    %out = affine.load %buf[0, 0] : memref<64x64xi32>
    memref.dealloc %buf : memref<64x64xi32>
    return %out : i32
  }
}
