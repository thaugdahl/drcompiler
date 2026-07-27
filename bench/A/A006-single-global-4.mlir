// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// A006: store to 4 globals, load each from different function.

module {
  memref.global "private" @g0 : memref<i32> = uninitialized
  memref.global "private" @g1 : memref<i32> = uninitialized
  memref.global "private" @g2 : memref<i32> = uninitialized
  memref.global "private" @g3 : memref<i32> = uninitialized

  func.func private @writer(%v0: i32, %v1: i32, %v2: i32, %v3: i32) {
    %g0 = memref.get_global @g0 : memref<i32>
    %g1 = memref.get_global @g1 : memref<i32>
    %g2 = memref.get_global @g2 : memref<i32>
    %g3 = memref.get_global @g3 : memref<i32>
    memref.store %v0, %g0[] : memref<i32>
    memref.store %v1, %g1[] : memref<i32>
    memref.store %v2, %g2[] : memref<i32>
    memref.store %v3, %g3[] : memref<i32>
    return
  }

  func.func private @reader() -> i32 {
    %g0 = memref.get_global @g0 : memref<i32>
    %g1 = memref.get_global @g1 : memref<i32>
    %g2 = memref.get_global @g2 : memref<i32>
    %g3 = memref.get_global @g3 : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v0 = memref.load %g0[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v1 = memref.load %g1[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v2 = memref.load %g2[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v3 = memref.load %g3[] : memref<i32>
    %s01 = arith.addi %v0, %v1 : i32
    %s23 = arith.addi %v2, %v3 : i32
    %sum = arith.addi %s01, %s23 : i32
    return %sum : i32
  }

  func.func @test(%a: i32, %b: i32, %c: i32, %d: i32) -> i32 {
    call @writer(%a, %b, %c, %d) : (i32, i32, i32, i32) -> ()
    %r = call @reader() : () -> i32
    return %r : i32
  }
}
