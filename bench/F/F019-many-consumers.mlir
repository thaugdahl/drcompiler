// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// F019: 1 store, 32 loads from the same buffer (sequential).
// Tests analysis with many consumer loads of a single store.

module {
  func.func @many_consumers(%arg0: i32) -> i32 {
    %buf = memref.alloc() : memref<i32>
    %c1 = arith.constant 1 : i32
    %val = arith.addi %arg0, %c1 : i32
    memref.store %val, %buf[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l0 = memref.load %buf[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l1 = memref.load %buf[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l2 = memref.load %buf[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l3 = memref.load %buf[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l4 = memref.load %buf[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l5 = memref.load %buf[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l6 = memref.load %buf[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l7 = memref.load %buf[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l8 = memref.load %buf[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l9 = memref.load %buf[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l10 = memref.load %buf[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l11 = memref.load %buf[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l12 = memref.load %buf[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l13 = memref.load %buf[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l14 = memref.load %buf[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l15 = memref.load %buf[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l16 = memref.load %buf[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l17 = memref.load %buf[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l18 = memref.load %buf[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l19 = memref.load %buf[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l20 = memref.load %buf[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l21 = memref.load %buf[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l22 = memref.load %buf[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l23 = memref.load %buf[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l24 = memref.load %buf[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l25 = memref.load %buf[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l26 = memref.load %buf[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l27 = memref.load %buf[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l28 = memref.load %buf[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l29 = memref.load %buf[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l30 = memref.load %buf[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l31 = memref.load %buf[] : memref<i32>
    %s0 = arith.addi %l0, %l1 : i32
    %s1 = arith.addi %s0, %l2 : i32
    %s2 = arith.addi %s1, %l3 : i32
    %s3 = arith.addi %s2, %l4 : i32
    %s4 = arith.addi %s3, %l5 : i32
    %s5 = arith.addi %s4, %l6 : i32
    %s6 = arith.addi %s5, %l7 : i32
    %s7 = arith.addi %s6, %l8 : i32
    %s8 = arith.addi %s7, %l9 : i32
    %s9 = arith.addi %s8, %l10 : i32
    %s10 = arith.addi %s9, %l11 : i32
    %s11 = arith.addi %s10, %l12 : i32
    %s12 = arith.addi %s11, %l13 : i32
    %s13 = arith.addi %s12, %l14 : i32
    %s14 = arith.addi %s13, %l15 : i32
    %s15 = arith.addi %s14, %l16 : i32
    %s16 = arith.addi %s15, %l17 : i32
    %s17 = arith.addi %s16, %l18 : i32
    %s18 = arith.addi %s17, %l19 : i32
    %s19 = arith.addi %s18, %l20 : i32
    %s20 = arith.addi %s19, %l21 : i32
    %s21 = arith.addi %s20, %l22 : i32
    %s22 = arith.addi %s21, %l23 : i32
    %s23 = arith.addi %s22, %l24 : i32
    %s24 = arith.addi %s23, %l25 : i32
    %s25 = arith.addi %s24, %l26 : i32
    %s26 = arith.addi %s25, %l27 : i32
    %s27 = arith.addi %s26, %l28 : i32
    %s28 = arith.addi %s27, %l29 : i32
    %s29 = arith.addi %s28, %l30 : i32
    %s30 = arith.addi %s29, %l31 : i32
    memref.dealloc %buf : memref<i32>
    return %s30 : i32
  }
}
