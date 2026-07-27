// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// F006: 16 independent alloc/store/load/dealloc sequences.
// Tests analysis scaling with 16 independent buffers.

module {
  func.func @buffers_16(%arg0: i32) -> i32 {
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
    %s0 = arith.addi %arg0, %c1 : i32
    memref.store %s0, %buf0[] : memref<i32>
    %s1 = arith.addi %s0, %c1 : i32
    memref.store %s1, %buf1[] : memref<i32>
    %s2 = arith.addi %s1, %c1 : i32
    memref.store %s2, %buf2[] : memref<i32>
    %s3 = arith.addi %s2, %c1 : i32
    memref.store %s3, %buf3[] : memref<i32>
    %s4 = arith.addi %s3, %c1 : i32
    memref.store %s4, %buf4[] : memref<i32>
    %s5 = arith.addi %s4, %c1 : i32
    memref.store %s5, %buf5[] : memref<i32>
    %s6 = arith.addi %s5, %c1 : i32
    memref.store %s6, %buf6[] : memref<i32>
    %s7 = arith.addi %s6, %c1 : i32
    memref.store %s7, %buf7[] : memref<i32>
    %s8 = arith.addi %s7, %c1 : i32
    memref.store %s8, %buf8[] : memref<i32>
    %s9 = arith.addi %s8, %c1 : i32
    memref.store %s9, %buf9[] : memref<i32>
    %s10 = arith.addi %s9, %c1 : i32
    memref.store %s10, %buf10[] : memref<i32>
    %s11 = arith.addi %s10, %c1 : i32
    memref.store %s11, %buf11[] : memref<i32>
    %s12 = arith.addi %s11, %c1 : i32
    memref.store %s12, %buf12[] : memref<i32>
    %s13 = arith.addi %s12, %c1 : i32
    memref.store %s13, %buf13[] : memref<i32>
    %s14 = arith.addi %s13, %c1 : i32
    memref.store %s14, %buf14[] : memref<i32>
    %s15 = arith.addi %s14, %c1 : i32
    memref.store %s15, %buf15[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l0 = memref.load %buf0[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l1 = memref.load %buf1[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l2 = memref.load %buf2[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l3 = memref.load %buf3[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l4 = memref.load %buf4[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l5 = memref.load %buf5[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l6 = memref.load %buf6[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l7 = memref.load %buf7[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l8 = memref.load %buf8[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l9 = memref.load %buf9[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l10 = memref.load %buf10[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l11 = memref.load %buf11[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l12 = memref.load %buf12[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l13 = memref.load %buf13[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l14 = memref.load %buf14[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l15 = memref.load %buf15[] : memref<i32>
    %r0 = arith.addi %l0, %l1 : i32
    %r1 = arith.addi %r0, %l2 : i32
    %r2 = arith.addi %r1, %l3 : i32
    %r3 = arith.addi %r2, %l4 : i32
    %r4 = arith.addi %r3, %l5 : i32
    %r5 = arith.addi %r4, %l6 : i32
    %r6 = arith.addi %r5, %l7 : i32
    %r7 = arith.addi %r6, %l8 : i32
    %r8 = arith.addi %r7, %l9 : i32
    %r9 = arith.addi %r8, %l10 : i32
    %r10 = arith.addi %r9, %l11 : i32
    %r11 = arith.addi %r10, %l12 : i32
    %r12 = arith.addi %r11, %l13 : i32
    %r13 = arith.addi %r12, %l14 : i32
    %r14 = arith.addi %r13, %l15 : i32
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
    return %r14 : i32
  }
}
