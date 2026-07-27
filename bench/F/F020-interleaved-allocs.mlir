// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// F020: 16 allocs with interleaved store/load patterns.
// Stores to even buffers, loads from even, stores to odd, loads from odd.

module {
  func.func @interleaved_allocs(%arg0: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %buf0 = memref.alloc() : memref<i32>
    %buf1 = memref.alloc() : memref<i32>
    %buf2 = memref.alloc() : memref<i32>
    %buf3 = memref.alloc() : memref<i32>
    %buf4 = memref.alloc() : memref<i32>
    %buf5 = memref.alloc() : memref<i32>
    %buf6 = memref.alloc() : memref<i32>
    %buf7 = memref.alloc() : memref<i32>
    %buf8 = memref.alloc() : memref<i32>
    %buf9 = memref.alloc() : memref<i32>
    %buf10 = memref.alloc() : memref<i32>
    %buf11 = memref.alloc() : memref<i32>
    %buf12 = memref.alloc() : memref<i32>
    %buf13 = memref.alloc() : memref<i32>
    %buf14 = memref.alloc() : memref<i32>
    %buf15 = memref.alloc() : memref<i32>
    %sv0 = arith.addi %arg0, %c1 : i32
    memref.store %sv0, %buf0[] : memref<i32>
    %sv2 = arith.addi %sv0, %c1 : i32
    memref.store %sv2, %buf2[] : memref<i32>
    %sv4 = arith.addi %sv2, %c1 : i32
    memref.store %sv4, %buf4[] : memref<i32>
    %sv6 = arith.addi %sv4, %c1 : i32
    memref.store %sv6, %buf6[] : memref<i32>
    %sv8 = arith.addi %sv6, %c1 : i32
    memref.store %sv8, %buf8[] : memref<i32>
    %sv10 = arith.addi %sv8, %c1 : i32
    memref.store %sv10, %buf10[] : memref<i32>
    %sv12 = arith.addi %sv10, %c1 : i32
    memref.store %sv12, %buf12[] : memref<i32>
    %sv14 = arith.addi %sv12, %c1 : i32
    memref.store %sv14, %buf14[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %le0 = memref.load %buf0[] : memref<i32>
    %sv1 = arith.addi %le0, %c1 : i32
    memref.store %sv1, %buf1[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %le2 = memref.load %buf2[] : memref<i32>
    %sv3 = arith.addi %le2, %c1 : i32
    memref.store %sv3, %buf3[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %le4 = memref.load %buf4[] : memref<i32>
    %sv5 = arith.addi %le4, %c1 : i32
    memref.store %sv5, %buf5[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %le6 = memref.load %buf6[] : memref<i32>
    %sv7 = arith.addi %le6, %c1 : i32
    memref.store %sv7, %buf7[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %le8 = memref.load %buf8[] : memref<i32>
    %sv9 = arith.addi %le8, %c1 : i32
    memref.store %sv9, %buf9[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %le10 = memref.load %buf10[] : memref<i32>
    %sv11 = arith.addi %le10, %c1 : i32
    memref.store %sv11, %buf11[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %le12 = memref.load %buf12[] : memref<i32>
    %sv13 = arith.addi %le12, %c1 : i32
    memref.store %sv13, %buf13[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %le14 = memref.load %buf14[] : memref<i32>
    %sv15 = arith.addi %le14, %c1 : i32
    memref.store %sv15, %buf15[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %lo1 = memref.load %buf1[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %lo3 = memref.load %buf3[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %lo5 = memref.load %buf5[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %lo7 = memref.load %buf7[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %lo9 = memref.load %buf9[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %lo11 = memref.load %buf11[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %lo13 = memref.load %buf13[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %lo15 = memref.load %buf15[] : memref<i32>
    %s0 = arith.addi %lo1, %lo3 : i32
    %s1 = arith.addi %s0, %lo5 : i32
    %s2 = arith.addi %s1, %lo7 : i32
    %s3 = arith.addi %s2, %lo9 : i32
    %s4 = arith.addi %s3, %lo11 : i32
    %s5 = arith.addi %s4, %lo13 : i32
    %s6 = arith.addi %s5, %lo15 : i32
    memref.dealloc %buf0 : memref<i32>
    memref.dealloc %buf1 : memref<i32>
    memref.dealloc %buf2 : memref<i32>
    memref.dealloc %buf3 : memref<i32>
    memref.dealloc %buf4 : memref<i32>
    memref.dealloc %buf5 : memref<i32>
    memref.dealloc %buf6 : memref<i32>
    memref.dealloc %buf7 : memref<i32>
    memref.dealloc %buf8 : memref<i32>
    memref.dealloc %buf9 : memref<i32>
    memref.dealloc %buf10 : memref<i32>
    memref.dealloc %buf11 : memref<i32>
    memref.dealloc %buf12 : memref<i32>
    memref.dealloc %buf13 : memref<i32>
    memref.dealloc %buf14 : memref<i32>
    memref.dealloc %buf15 : memref<i32>
    return %s6 : i32
  }
}
