// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// F001: 16-op linear add chain, store result, load elsewhere.
// Tests that the analysis handles moderate chain depth correctly.

module {
  func.func @chain_depth_16(%arg0: i32) -> i32 {
    %buf = memref.alloc() : memref<i32>
    %c1 = arith.constant 1 : i32
    %v1 = arith.addi %arg0, %c1 : i32
    %v2 = arith.addi %v1, %c1 : i32
    %v3 = arith.addi %v2, %c1 : i32
    %v4 = arith.addi %v3, %c1 : i32
    %v5 = arith.addi %v4, %c1 : i32
    %v6 = arith.addi %v5, %c1 : i32
    %v7 = arith.addi %v6, %c1 : i32
    %v8 = arith.addi %v7, %c1 : i32
    %v9 = arith.addi %v8, %c1 : i32
    %v10 = arith.addi %v9, %c1 : i32
    %v11 = arith.addi %v10, %c1 : i32
    %v12 = arith.addi %v11, %c1 : i32
    %v13 = arith.addi %v12, %c1 : i32
    %v14 = arith.addi %v13, %c1 : i32
    %v15 = arith.addi %v14, %c1 : i32
    %v16 = arith.addi %v15, %c1 : i32
    memref.store %v16, %buf[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %out = memref.load %buf[] : memref<i32>
    memref.dealloc %buf : memref<i32>
    return %out : i32
  }
}
