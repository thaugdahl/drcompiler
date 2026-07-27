// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// A014: memref escapes because callee forwards it to an external function
// AND returns it. The return is the escape vector distinguishing this from
// a simple external-call leak.

module {
  func.func private @extern_process(%m: memref<i32>) -> memref<i32>

  func.func @leaked_return() {
    %alloc = memref.alloc() : memref<i32>
    %c42 = arith.constant 42 : i32
    memref.store %c42, %alloc[] : memref<i32>
    %ret = call @extern_process(%alloc) : (memref<i32>) -> memref<i32>
    // The external function may have written to the buffer.
    // expected-remark @below {{load: LEAKED}}
    %v = memref.load %alloc[] : memref<i32>
    memref.dealloc %alloc : memref<i32>
    return
  }
}
