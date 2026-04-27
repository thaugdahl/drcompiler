// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// S3-D: a defined callee performs an unconditional rank-0 store through its
// memref argument.  The caller stores 42, calls the callee (which writes
// 99), then loads.  Without the fix, the prior caller store and the callee
// store both reach the load -> MULTI.  With the rank-0 kill semantics
// applied across the call boundary, the callee's unconditional rank-0
// overwrite kills the caller's prior store, leaving SINGLE provenance from
// the callee.

module {
  func.func private @callee_writes_99(%out: memref<i32>) {
    %c99 = arith.constant 99 : i32
    memref.store %c99, %out[] : memref<i32>
    return
  }

  func.func @test() -> i32 {
    %a = memref.alloc() : memref<i32>
    %c42 = arith.constant 42 : i32
    memref.store %c42, %a[] : memref<i32>
    call @callee_writes_99(%a) : (memref<i32>) -> ()
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %a[] : memref<i32>
    memref.dealloc %a : memref<i32>
    return %v : i32
  }
}
