// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// F011: linear call chain of depth 4.
// func0 -> func1 -> ... -> func3, each stores to a global, last reader loads.

module {
  memref.global "private" @g : memref<i32> = uninitialized

  func.func private @func0(%x: i32) {
    %g = memref.get_global @g : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func private @func1(%x: i32) {
    %c1 = arith.constant 1 : i32
    %v = arith.addi %x, %c1 : i32
    call @func0(%v) : (i32) -> ()
    return
  }

  func.func private @func2(%x: i32) {
    %c1 = arith.constant 1 : i32
    %v = arith.addi %x, %c1 : i32
    call @func1(%v) : (i32) -> ()
    return
  }

  func.func private @func3(%x: i32) {
    %c1 = arith.constant 1 : i32
    %v = arith.addi %x, %c1 : i32
    call @func2(%v) : (i32) -> ()
    return
  }

  func.func private @reader() -> i32 {
    %g = memref.get_global @g : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func @main(%x: i32) -> i32 {
    call @func3(%x) : (i32) -> ()
    %r = call @reader() : () -> i32
    return %r : i32
  }
}
