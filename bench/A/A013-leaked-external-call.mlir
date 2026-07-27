// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// A013: memref passed to external (body-less) function -> LEAKED.

module {
  func.func private @external_func(%arg: memref<i32>)

  func.func @leaked_external_call() {
    %alloc = memref.alloc() : memref<i32>
    %c42 = arith.constant 42 : i32
    memref.store %c42, %alloc[] : memref<i32>
    call @external_func(%alloc) : (memref<i32>) -> ()
    // expected-remark @below {{load: LEAKED}}
    %v = memref.load %alloc[] : memref<i32>
    memref.dealloc %alloc : memref<i32>
    return
  }
}
