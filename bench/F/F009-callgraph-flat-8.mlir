// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// F009: main calls 8 leaf functions, each stores to own global.
// Tests call-graph analysis scaling with 8 flat callees.

module {
  memref.global "private" @g0 : memref<i32> = uninitialized
  memref.global "private" @g1 : memref<i32> = uninitialized
  memref.global "private" @g2 : memref<i32> = uninitialized
  memref.global "private" @g3 : memref<i32> = uninitialized
  memref.global "private" @g4 : memref<i32> = uninitialized
  memref.global "private" @g5 : memref<i32> = uninitialized
  memref.global "private" @g6 : memref<i32> = uninitialized
  memref.global "private" @g7 : memref<i32> = uninitialized

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

  func.func @main(%x: i32) -> i32 {
    call @leaf0(%x) : (i32) -> ()
    call @leaf1(%x) : (i32) -> ()
    call @leaf2(%x) : (i32) -> ()
    call @leaf3(%x) : (i32) -> ()
    call @leaf4(%x) : (i32) -> ()
    call @leaf5(%x) : (i32) -> ()
    call @leaf6(%x) : (i32) -> ()
    call @leaf7(%x) : (i32) -> ()
    %r0 = call @reader0() : () -> i32
    %r1 = call @reader1() : () -> i32
    %r2 = call @reader2() : () -> i32
    %r3 = call @reader3() : () -> i32
    %r4 = call @reader4() : () -> i32
    %r5 = call @reader5() : () -> i32
    %r6 = call @reader6() : () -> i32
    %r7 = call @reader7() : () -> i32
    %s0 = arith.addi %r0, %r1 : i32
    %s1 = arith.addi %s0, %r2 : i32
    %s2 = arith.addi %s1, %r3 : i32
    %s3 = arith.addi %s2, %r4 : i32
    %s4 = arith.addi %s3, %r5 : i32
    %s5 = arith.addi %s4, %r6 : i32
    %s6 = arith.addi %s5, %r7 : i32
    return %s6 : i32
  }
}
