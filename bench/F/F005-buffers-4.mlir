// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// F005: 4 independent alloc/store/load/dealloc sequences.
// Tests analysis scaling with 4 independent buffers.

module {
  func.func @buffers_4(%arg0: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %buf0 = memref.alloc() : memref<i32>
    %buf1 = memref.alloc() : memref<i32>
    %buf2 = memref.alloc() : memref<i32>
    %buf3 = memref.alloc() : memref<i32>
    %s0 = arith.addi %arg0, %c1 : i32
    memref.store %s0, %buf0[] : memref<i32>
    %s1 = arith.addi %s0, %c1 : i32
    memref.store %s1, %buf1[] : memref<i32>
    %s2 = arith.addi %s1, %c1 : i32
    memref.store %s2, %buf2[] : memref<i32>
    %s3 = arith.addi %s2, %c1 : i32
    memref.store %s3, %buf3[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l0 = memref.load %buf0[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l1 = memref.load %buf1[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l2 = memref.load %buf2[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l3 = memref.load %buf3[] : memref<i32>
    %r0 = arith.addi %l0, %l1 : i32
    %r1 = arith.addi %r0, %l2 : i32
    %r2 = arith.addi %r1, %l3 : i32
    memref.dealloc %buf0 : memref<i32>
    memref.dealloc %buf1 : memref<i32>
    memref.dealloc %buf2 : memref<i32>
    memref.dealloc %buf3 : memref<i32>
    return %r2 : i32
  }
}
