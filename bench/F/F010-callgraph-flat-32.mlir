// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// F010: main calls 32 leaf functions, each stores to own global.
// Tests call-graph analysis scaling with 32 flat callees.

module {
  memref.global "private" @g0 : memref<i32> = uninitialized
  memref.global "private" @g1 : memref<i32> = uninitialized
  memref.global "private" @g2 : memref<i32> = uninitialized
  memref.global "private" @g3 : memref<i32> = uninitialized
  memref.global "private" @g4 : memref<i32> = uninitialized
  memref.global "private" @g5 : memref<i32> = uninitialized
  memref.global "private" @g6 : memref<i32> = uninitialized
  memref.global "private" @g7 : memref<i32> = uninitialized
  memref.global "private" @g8 : memref<i32> = uninitialized
  memref.global "private" @g9 : memref<i32> = uninitialized
  memref.global "private" @g10 : memref<i32> = uninitialized
  memref.global "private" @g11 : memref<i32> = uninitialized
  memref.global "private" @g12 : memref<i32> = uninitialized
  memref.global "private" @g13 : memref<i32> = uninitialized
  memref.global "private" @g14 : memref<i32> = uninitialized
  memref.global "private" @g15 : memref<i32> = uninitialized
  memref.global "private" @g16 : memref<i32> = uninitialized
  memref.global "private" @g17 : memref<i32> = uninitialized
  memref.global "private" @g18 : memref<i32> = uninitialized
  memref.global "private" @g19 : memref<i32> = uninitialized
  memref.global "private" @g20 : memref<i32> = uninitialized
  memref.global "private" @g21 : memref<i32> = uninitialized
  memref.global "private" @g22 : memref<i32> = uninitialized
  memref.global "private" @g23 : memref<i32> = uninitialized
  memref.global "private" @g24 : memref<i32> = uninitialized
  memref.global "private" @g25 : memref<i32> = uninitialized
  memref.global "private" @g26 : memref<i32> = uninitialized
  memref.global "private" @g27 : memref<i32> = uninitialized
  memref.global "private" @g28 : memref<i32> = uninitialized
  memref.global "private" @g29 : memref<i32> = uninitialized
  memref.global "private" @g30 : memref<i32> = uninitialized
  memref.global "private" @g31 : memref<i32> = uninitialized

  func.func private @leaf0(%x: i32) {
    %g = memref.get_global @g0 : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @leaf1(%x: i32) {
    %g = memref.get_global @g1 : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @leaf2(%x: i32) {
    %g = memref.get_global @g2 : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @leaf3(%x: i32) {
    %g = memref.get_global @g3 : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @leaf4(%x: i32) {
    %g = memref.get_global @g4 : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @leaf5(%x: i32) {
    %g = memref.get_global @g5 : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @leaf6(%x: i32) {
    %g = memref.get_global @g6 : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @leaf7(%x: i32) {
    %g = memref.get_global @g7 : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @leaf8(%x: i32) {
    %g = memref.get_global @g8 : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @leaf9(%x: i32) {
    %g = memref.get_global @g9 : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @leaf10(%x: i32) {
    %g = memref.get_global @g10 : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @leaf11(%x: i32) {
    %g = memref.get_global @g11 : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @leaf12(%x: i32) {
    %g = memref.get_global @g12 : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @leaf13(%x: i32) {
    %g = memref.get_global @g13 : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @leaf14(%x: i32) {
    %g = memref.get_global @g14 : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @leaf15(%x: i32) {
    %g = memref.get_global @g15 : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @leaf16(%x: i32) {
    %g = memref.get_global @g16 : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @leaf17(%x: i32) {
    %g = memref.get_global @g17 : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @leaf18(%x: i32) {
    %g = memref.get_global @g18 : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @leaf19(%x: i32) {
    %g = memref.get_global @g19 : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @leaf20(%x: i32) {
    %g = memref.get_global @g20 : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @leaf21(%x: i32) {
    %g = memref.get_global @g21 : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @leaf22(%x: i32) {
    %g = memref.get_global @g22 : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @leaf23(%x: i32) {
    %g = memref.get_global @g23 : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @leaf24(%x: i32) {
    %g = memref.get_global @g24 : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @leaf25(%x: i32) {
    %g = memref.get_global @g25 : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @leaf26(%x: i32) {
    %g = memref.get_global @g26 : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @leaf27(%x: i32) {
    %g = memref.get_global @g27 : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @leaf28(%x: i32) {
    %g = memref.get_global @g28 : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @leaf29(%x: i32) {
    %g = memref.get_global @g29 : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @leaf30(%x: i32) {
    %g = memref.get_global @g30 : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @leaf31(%x: i32) {
    %g = memref.get_global @g31 : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @reader0() -> i32 {
    %g = memref.get_global @g0 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func private @reader1() -> i32 {
    %g = memref.get_global @g1 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func private @reader2() -> i32 {
    %g = memref.get_global @g2 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func private @reader3() -> i32 {
    %g = memref.get_global @g3 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func private @reader4() -> i32 {
    %g = memref.get_global @g4 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func private @reader5() -> i32 {
    %g = memref.get_global @g5 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func private @reader6() -> i32 {
    %g = memref.get_global @g6 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func private @reader7() -> i32 {
    %g = memref.get_global @g7 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func private @reader8() -> i32 {
    %g = memref.get_global @g8 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func private @reader9() -> i32 {
    %g = memref.get_global @g9 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func private @reader10() -> i32 {
    %g = memref.get_global @g10 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func private @reader11() -> i32 {
    %g = memref.get_global @g11 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func private @reader12() -> i32 {
    %g = memref.get_global @g12 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func private @reader13() -> i32 {
    %g = memref.get_global @g13 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func private @reader14() -> i32 {
    %g = memref.get_global @g14 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func private @reader15() -> i32 {
    %g = memref.get_global @g15 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func private @reader16() -> i32 {
    %g = memref.get_global @g16 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func private @reader17() -> i32 {
    %g = memref.get_global @g17 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func private @reader18() -> i32 {
    %g = memref.get_global @g18 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func private @reader19() -> i32 {
    %g = memref.get_global @g19 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func private @reader20() -> i32 {
    %g = memref.get_global @g20 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func private @reader21() -> i32 {
    %g = memref.get_global @g21 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func private @reader22() -> i32 {
    %g = memref.get_global @g22 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func private @reader23() -> i32 {
    %g = memref.get_global @g23 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func private @reader24() -> i32 {
    %g = memref.get_global @g24 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func private @reader25() -> i32 {
    %g = memref.get_global @g25 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func private @reader26() -> i32 {
    %g = memref.get_global @g26 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func private @reader27() -> i32 {
    %g = memref.get_global @g27 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func private @reader28() -> i32 {
    %g = memref.get_global @g28 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func private @reader29() -> i32 {
    %g = memref.get_global @g29 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func private @reader30() -> i32 {
    %g = memref.get_global @g30 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func private @reader31() -> i32 {
    %g = memref.get_global @g31 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func @main(%x: i32) -> i32 {
    call @leaf0(%x) : (i32) -> ()
    call @leaf1(%x) : (i32) -> ()
    call @leaf2(%x) : (i32) -> ()
    call @leaf3(%x) : (i32) -> ()
    call @leaf4(%x) : (i32) -> ()
    call @leaf5(%x) : (i32) -> ()
    call @leaf6(%x) : (i32) -> ()
    call @leaf7(%x) : (i32) -> ()
    call @leaf8(%x) : (i32) -> ()
    call @leaf9(%x) : (i32) -> ()
    call @leaf10(%x) : (i32) -> ()
    call @leaf11(%x) : (i32) -> ()
    call @leaf12(%x) : (i32) -> ()
    call @leaf13(%x) : (i32) -> ()
    call @leaf14(%x) : (i32) -> ()
    call @leaf15(%x) : (i32) -> ()
    call @leaf16(%x) : (i32) -> ()
    call @leaf17(%x) : (i32) -> ()
    call @leaf18(%x) : (i32) -> ()
    call @leaf19(%x) : (i32) -> ()
    call @leaf20(%x) : (i32) -> ()
    call @leaf21(%x) : (i32) -> ()
    call @leaf22(%x) : (i32) -> ()
    call @leaf23(%x) : (i32) -> ()
    call @leaf24(%x) : (i32) -> ()
    call @leaf25(%x) : (i32) -> ()
    call @leaf26(%x) : (i32) -> ()
    call @leaf27(%x) : (i32) -> ()
    call @leaf28(%x) : (i32) -> ()
    call @leaf29(%x) : (i32) -> ()
    call @leaf30(%x) : (i32) -> ()
    call @leaf31(%x) : (i32) -> ()
    %r0 = call @reader0() : () -> i32
    %r1 = call @reader1() : () -> i32
    %r2 = call @reader2() : () -> i32
    %r3 = call @reader3() : () -> i32
    %r4 = call @reader4() : () -> i32
    %r5 = call @reader5() : () -> i32
    %r6 = call @reader6() : () -> i32
    %r7 = call @reader7() : () -> i32
    %r8 = call @reader8() : () -> i32
    %r9 = call @reader9() : () -> i32
    %r10 = call @reader10() : () -> i32
    %r11 = call @reader11() : () -> i32
    %r12 = call @reader12() : () -> i32
    %r13 = call @reader13() : () -> i32
    %r14 = call @reader14() : () -> i32
    %r15 = call @reader15() : () -> i32
    %r16 = call @reader16() : () -> i32
    %r17 = call @reader17() : () -> i32
    %r18 = call @reader18() : () -> i32
    %r19 = call @reader19() : () -> i32
    %r20 = call @reader20() : () -> i32
    %r21 = call @reader21() : () -> i32
    %r22 = call @reader22() : () -> i32
    %r23 = call @reader23() : () -> i32
    %r24 = call @reader24() : () -> i32
    %r25 = call @reader25() : () -> i32
    %r26 = call @reader26() : () -> i32
    %r27 = call @reader27() : () -> i32
    %r28 = call @reader28() : () -> i32
    %r29 = call @reader29() : () -> i32
    %r30 = call @reader30() : () -> i32
    %r31 = call @reader31() : () -> i32
    %s0 = arith.addi %r0, %r1 : i32
    %s1 = arith.addi %s0, %r2 : i32
    %s2 = arith.addi %s1, %r3 : i32
    %s3 = arith.addi %s2, %r4 : i32
    %s4 = arith.addi %s3, %r5 : i32
    %s5 = arith.addi %s4, %r6 : i32
    %s6 = arith.addi %s5, %r7 : i32
    %s7 = arith.addi %s6, %r8 : i32
    %s8 = arith.addi %s7, %r9 : i32
    %s9 = arith.addi %s8, %r10 : i32
    %s10 = arith.addi %s9, %r11 : i32
    %s11 = arith.addi %s10, %r12 : i32
    %s12 = arith.addi %s11, %r13 : i32
    %s13 = arith.addi %s12, %r14 : i32
    %s14 = arith.addi %s13, %r15 : i32
    %s15 = arith.addi %s14, %r16 : i32
    %s16 = arith.addi %s15, %r17 : i32
    %s17 = arith.addi %s16, %r18 : i32
    %s18 = arith.addi %s17, %r19 : i32
    %s19 = arith.addi %s18, %r20 : i32
    %s20 = arith.addi %s19, %r21 : i32
    %s21 = arith.addi %s20, %r22 : i32
    %s22 = arith.addi %s21, %r23 : i32
    %s23 = arith.addi %s22, %r24 : i32
    %s24 = arith.addi %s23, %r25 : i32
    %s25 = arith.addi %s24, %r26 : i32
    %s26 = arith.addi %s25, %r27 : i32
    %s27 = arith.addi %s26, %r28 : i32
    %s28 = arith.addi %s27, %r29 : i32
    %s29 = arith.addi %s28, %r30 : i32
    %s30 = arith.addi %s29, %r31 : i32
    return %s30 : i32
  }
}
