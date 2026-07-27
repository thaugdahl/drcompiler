// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// A017: memref escapes through 2-hop call chain to external function.
// hop1 calls hop2, hop2 is external. Memref is passed through.

module {
  func.func private @external_sink(%m: memref<i32>)

  func.func private @hop1(%m: memref<i32>) {
    call @external_sink(%m) : (memref<i32>) -> ()
    return
  }

  func.func @leaked_nested_call() {
    %alloc = memref.alloc() : memref<i32>
    %c42 = arith.constant 42 : i32
    memref.store %c42, %alloc[] : memref<i32>
    call @hop1(%alloc) : (memref<i32>) -> ()
    // expected-remark @below {{load: LEAKED}}
    %v = memref.load %alloc[] : memref<i32>
    memref.dealloc %alloc : memref<i32>
    return
  }
}
