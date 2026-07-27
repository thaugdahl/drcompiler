// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// A005a: store in callee, load in caller via global (depth 1).

module {
  memref.global "private" @g : memref<i32> = uninitialized

  func.func private @writer(%x: i32) {
    %g = memref.get_global @g : memref<i32>
    memref.store %x, %g[] : memref<i32>
    return
  }

  func.func @test(%x: i32) -> i32 {
    call @writer(%x) : (i32) -> ()
    %g = memref.get_global @g : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }
}
