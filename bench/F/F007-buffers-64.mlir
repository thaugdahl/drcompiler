// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// F007: 64 independent alloc/store/load/dealloc sequences.
// Tests analysis scaling with 64 independent buffers.

module {
  func.func @buffers_64(%arg0: i32) -> i32 {
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
    %buf16 = memref.alloc() : memref<i32>
    %buf17 = memref.alloc() : memref<i32>
    %buf18 = memref.alloc() : memref<i32>
    %buf19 = memref.alloc() : memref<i32>
    %buf20 = memref.alloc() : memref<i32>
    %buf21 = memref.alloc() : memref<i32>
    %buf22 = memref.alloc() : memref<i32>
    %buf23 = memref.alloc() : memref<i32>
    %buf24 = memref.alloc() : memref<i32>
    %buf25 = memref.alloc() : memref<i32>
    %buf26 = memref.alloc() : memref<i32>
    %buf27 = memref.alloc() : memref<i32>
    %buf28 = memref.alloc() : memref<i32>
    %buf29 = memref.alloc() : memref<i32>
    %buf30 = memref.alloc() : memref<i32>
    %buf31 = memref.alloc() : memref<i32>
    %buf32 = memref.alloc() : memref<i32>
    %buf33 = memref.alloc() : memref<i32>
    %buf34 = memref.alloc() : memref<i32>
    %buf35 = memref.alloc() : memref<i32>
    %buf36 = memref.alloc() : memref<i32>
    %buf37 = memref.alloc() : memref<i32>
    %buf38 = memref.alloc() : memref<i32>
    %buf39 = memref.alloc() : memref<i32>
    %buf40 = memref.alloc() : memref<i32>
    %buf41 = memref.alloc() : memref<i32>
    %buf42 = memref.alloc() : memref<i32>
    %buf43 = memref.alloc() : memref<i32>
    %buf44 = memref.alloc() : memref<i32>
    %buf45 = memref.alloc() : memref<i32>
    %buf46 = memref.alloc() : memref<i32>
    %buf47 = memref.alloc() : memref<i32>
    %buf48 = memref.alloc() : memref<i32>
    %buf49 = memref.alloc() : memref<i32>
    %buf50 = memref.alloc() : memref<i32>
    %buf51 = memref.alloc() : memref<i32>
    %buf52 = memref.alloc() : memref<i32>
    %buf53 = memref.alloc() : memref<i32>
    %buf54 = memref.alloc() : memref<i32>
    %buf55 = memref.alloc() : memref<i32>
    %buf56 = memref.alloc() : memref<i32>
    %buf57 = memref.alloc() : memref<i32>
    %buf58 = memref.alloc() : memref<i32>
    %buf59 = memref.alloc() : memref<i32>
    %buf60 = memref.alloc() : memref<i32>
    %buf61 = memref.alloc() : memref<i32>
    %buf62 = memref.alloc() : memref<i32>
    %buf63 = memref.alloc() : memref<i32>
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
    %s16 = arith.addi %s15, %c1 : i32
    memref.store %s16, %buf16[] : memref<i32>
    %s17 = arith.addi %s16, %c1 : i32
    memref.store %s17, %buf17[] : memref<i32>
    %s18 = arith.addi %s17, %c1 : i32
    memref.store %s18, %buf18[] : memref<i32>
    %s19 = arith.addi %s18, %c1 : i32
    memref.store %s19, %buf19[] : memref<i32>
    %s20 = arith.addi %s19, %c1 : i32
    memref.store %s20, %buf20[] : memref<i32>
    %s21 = arith.addi %s20, %c1 : i32
    memref.store %s21, %buf21[] : memref<i32>
    %s22 = arith.addi %s21, %c1 : i32
    memref.store %s22, %buf22[] : memref<i32>
    %s23 = arith.addi %s22, %c1 : i32
    memref.store %s23, %buf23[] : memref<i32>
    %s24 = arith.addi %s23, %c1 : i32
    memref.store %s24, %buf24[] : memref<i32>
    %s25 = arith.addi %s24, %c1 : i32
    memref.store %s25, %buf25[] : memref<i32>
    %s26 = arith.addi %s25, %c1 : i32
    memref.store %s26, %buf26[] : memref<i32>
    %s27 = arith.addi %s26, %c1 : i32
    memref.store %s27, %buf27[] : memref<i32>
    %s28 = arith.addi %s27, %c1 : i32
    memref.store %s28, %buf28[] : memref<i32>
    %s29 = arith.addi %s28, %c1 : i32
    memref.store %s29, %buf29[] : memref<i32>
    %s30 = arith.addi %s29, %c1 : i32
    memref.store %s30, %buf30[] : memref<i32>
    %s31 = arith.addi %s30, %c1 : i32
    memref.store %s31, %buf31[] : memref<i32>
    %s32 = arith.addi %s31, %c1 : i32
    memref.store %s32, %buf32[] : memref<i32>
    %s33 = arith.addi %s32, %c1 : i32
    memref.store %s33, %buf33[] : memref<i32>
    %s34 = arith.addi %s33, %c1 : i32
    memref.store %s34, %buf34[] : memref<i32>
    %s35 = arith.addi %s34, %c1 : i32
    memref.store %s35, %buf35[] : memref<i32>
    %s36 = arith.addi %s35, %c1 : i32
    memref.store %s36, %buf36[] : memref<i32>
    %s37 = arith.addi %s36, %c1 : i32
    memref.store %s37, %buf37[] : memref<i32>
    %s38 = arith.addi %s37, %c1 : i32
    memref.store %s38, %buf38[] : memref<i32>
    %s39 = arith.addi %s38, %c1 : i32
    memref.store %s39, %buf39[] : memref<i32>
    %s40 = arith.addi %s39, %c1 : i32
    memref.store %s40, %buf40[] : memref<i32>
    %s41 = arith.addi %s40, %c1 : i32
    memref.store %s41, %buf41[] : memref<i32>
    %s42 = arith.addi %s41, %c1 : i32
    memref.store %s42, %buf42[] : memref<i32>
    %s43 = arith.addi %s42, %c1 : i32
    memref.store %s43, %buf43[] : memref<i32>
    %s44 = arith.addi %s43, %c1 : i32
    memref.store %s44, %buf44[] : memref<i32>
    %s45 = arith.addi %s44, %c1 : i32
    memref.store %s45, %buf45[] : memref<i32>
    %s46 = arith.addi %s45, %c1 : i32
    memref.store %s46, %buf46[] : memref<i32>
    %s47 = arith.addi %s46, %c1 : i32
    memref.store %s47, %buf47[] : memref<i32>
    %s48 = arith.addi %s47, %c1 : i32
    memref.store %s48, %buf48[] : memref<i32>
    %s49 = arith.addi %s48, %c1 : i32
    memref.store %s49, %buf49[] : memref<i32>
    %s50 = arith.addi %s49, %c1 : i32
    memref.store %s50, %buf50[] : memref<i32>
    %s51 = arith.addi %s50, %c1 : i32
    memref.store %s51, %buf51[] : memref<i32>
    %s52 = arith.addi %s51, %c1 : i32
    memref.store %s52, %buf52[] : memref<i32>
    %s53 = arith.addi %s52, %c1 : i32
    memref.store %s53, %buf53[] : memref<i32>
    %s54 = arith.addi %s53, %c1 : i32
    memref.store %s54, %buf54[] : memref<i32>
    %s55 = arith.addi %s54, %c1 : i32
    memref.store %s55, %buf55[] : memref<i32>
    %s56 = arith.addi %s55, %c1 : i32
    memref.store %s56, %buf56[] : memref<i32>
    %s57 = arith.addi %s56, %c1 : i32
    memref.store %s57, %buf57[] : memref<i32>
    %s58 = arith.addi %s57, %c1 : i32
    memref.store %s58, %buf58[] : memref<i32>
    %s59 = arith.addi %s58, %c1 : i32
    memref.store %s59, %buf59[] : memref<i32>
    %s60 = arith.addi %s59, %c1 : i32
    memref.store %s60, %buf60[] : memref<i32>
    %s61 = arith.addi %s60, %c1 : i32
    memref.store %s61, %buf61[] : memref<i32>
    %s62 = arith.addi %s61, %c1 : i32
    memref.store %s62, %buf62[] : memref<i32>
    %s63 = arith.addi %s62, %c1 : i32
    memref.store %s63, %buf63[] : memref<i32>
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
    // expected-remark @below {{load: SINGLE}}
    %l16 = memref.load %buf16[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l17 = memref.load %buf17[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l18 = memref.load %buf18[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l19 = memref.load %buf19[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l20 = memref.load %buf20[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l21 = memref.load %buf21[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l22 = memref.load %buf22[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l23 = memref.load %buf23[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l24 = memref.load %buf24[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l25 = memref.load %buf25[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l26 = memref.load %buf26[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l27 = memref.load %buf27[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l28 = memref.load %buf28[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l29 = memref.load %buf29[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l30 = memref.load %buf30[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l31 = memref.load %buf31[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l32 = memref.load %buf32[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l33 = memref.load %buf33[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l34 = memref.load %buf34[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l35 = memref.load %buf35[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l36 = memref.load %buf36[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l37 = memref.load %buf37[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l38 = memref.load %buf38[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l39 = memref.load %buf39[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l40 = memref.load %buf40[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l41 = memref.load %buf41[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l42 = memref.load %buf42[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l43 = memref.load %buf43[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l44 = memref.load %buf44[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l45 = memref.load %buf45[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l46 = memref.load %buf46[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l47 = memref.load %buf47[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l48 = memref.load %buf48[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l49 = memref.load %buf49[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l50 = memref.load %buf50[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l51 = memref.load %buf51[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l52 = memref.load %buf52[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l53 = memref.load %buf53[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l54 = memref.load %buf54[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l55 = memref.load %buf55[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l56 = memref.load %buf56[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l57 = memref.load %buf57[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l58 = memref.load %buf58[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l59 = memref.load %buf59[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l60 = memref.load %buf60[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l61 = memref.load %buf61[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l62 = memref.load %buf62[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %l63 = memref.load %buf63[] : memref<i32>
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
    %r15 = arith.addi %r14, %l16 : i32
    %r16 = arith.addi %r15, %l17 : i32
    %r17 = arith.addi %r16, %l18 : i32
    %r18 = arith.addi %r17, %l19 : i32
    %r19 = arith.addi %r18, %l20 : i32
    %r20 = arith.addi %r19, %l21 : i32
    %r21 = arith.addi %r20, %l22 : i32
    %r22 = arith.addi %r21, %l23 : i32
    %r23 = arith.addi %r22, %l24 : i32
    %r24 = arith.addi %r23, %l25 : i32
    %r25 = arith.addi %r24, %l26 : i32
    %r26 = arith.addi %r25, %l27 : i32
    %r27 = arith.addi %r26, %l28 : i32
    %r28 = arith.addi %r27, %l29 : i32
    %r29 = arith.addi %r28, %l30 : i32
    %r30 = arith.addi %r29, %l31 : i32
    %r31 = arith.addi %r30, %l32 : i32
    %r32 = arith.addi %r31, %l33 : i32
    %r33 = arith.addi %r32, %l34 : i32
    %r34 = arith.addi %r33, %l35 : i32
    %r35 = arith.addi %r34, %l36 : i32
    %r36 = arith.addi %r35, %l37 : i32
    %r37 = arith.addi %r36, %l38 : i32
    %r38 = arith.addi %r37, %l39 : i32
    %r39 = arith.addi %r38, %l40 : i32
    %r40 = arith.addi %r39, %l41 : i32
    %r41 = arith.addi %r40, %l42 : i32
    %r42 = arith.addi %r41, %l43 : i32
    %r43 = arith.addi %r42, %l44 : i32
    %r44 = arith.addi %r43, %l45 : i32
    %r45 = arith.addi %r44, %l46 : i32
    %r46 = arith.addi %r45, %l47 : i32
    %r47 = arith.addi %r46, %l48 : i32
    %r48 = arith.addi %r47, %l49 : i32
    %r49 = arith.addi %r48, %l50 : i32
    %r50 = arith.addi %r49, %l51 : i32
    %r51 = arith.addi %r50, %l52 : i32
    %r52 = arith.addi %r51, %l53 : i32
    %r53 = arith.addi %r52, %l54 : i32
    %r54 = arith.addi %r53, %l55 : i32
    %r55 = arith.addi %r54, %l56 : i32
    %r56 = arith.addi %r55, %l57 : i32
    %r57 = arith.addi %r56, %l58 : i32
    %r58 = arith.addi %r57, %l59 : i32
    %r59 = arith.addi %r58, %l60 : i32
    %r60 = arith.addi %r59, %l61 : i32
    %r61 = arith.addi %r60, %l62 : i32
    %r62 = arith.addi %r61, %l63 : i32
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
    memref.dealloc %buf16 : memref<i32>
    memref.dealloc %buf17 : memref<i32>
    memref.dealloc %buf18 : memref<i32>
    memref.dealloc %buf19 : memref<i32>
    memref.dealloc %buf20 : memref<i32>
    memref.dealloc %buf21 : memref<i32>
    memref.dealloc %buf22 : memref<i32>
    memref.dealloc %buf23 : memref<i32>
    memref.dealloc %buf24 : memref<i32>
    memref.dealloc %buf25 : memref<i32>
    memref.dealloc %buf26 : memref<i32>
    memref.dealloc %buf27 : memref<i32>
    memref.dealloc %buf28 : memref<i32>
    memref.dealloc %buf29 : memref<i32>
    memref.dealloc %buf30 : memref<i32>
    memref.dealloc %buf31 : memref<i32>
    memref.dealloc %buf32 : memref<i32>
    memref.dealloc %buf33 : memref<i32>
    memref.dealloc %buf34 : memref<i32>
    memref.dealloc %buf35 : memref<i32>
    memref.dealloc %buf36 : memref<i32>
    memref.dealloc %buf37 : memref<i32>
    memref.dealloc %buf38 : memref<i32>
    memref.dealloc %buf39 : memref<i32>
    memref.dealloc %buf40 : memref<i32>
    memref.dealloc %buf41 : memref<i32>
    memref.dealloc %buf42 : memref<i32>
    memref.dealloc %buf43 : memref<i32>
    memref.dealloc %buf44 : memref<i32>
    memref.dealloc %buf45 : memref<i32>
    memref.dealloc %buf46 : memref<i32>
    memref.dealloc %buf47 : memref<i32>
    memref.dealloc %buf48 : memref<i32>
    memref.dealloc %buf49 : memref<i32>
    memref.dealloc %buf50 : memref<i32>
    memref.dealloc %buf51 : memref<i32>
    memref.dealloc %buf52 : memref<i32>
    memref.dealloc %buf53 : memref<i32>
    memref.dealloc %buf54 : memref<i32>
    memref.dealloc %buf55 : memref<i32>
    memref.dealloc %buf56 : memref<i32>
    memref.dealloc %buf57 : memref<i32>
    memref.dealloc %buf58 : memref<i32>
    memref.dealloc %buf59 : memref<i32>
    memref.dealloc %buf60 : memref<i32>
    memref.dealloc %buf61 : memref<i32>
    memref.dealloc %buf62 : memref<i32>
    memref.dealloc %buf63 : memref<i32>
    return %r62 : i32
  }
}
