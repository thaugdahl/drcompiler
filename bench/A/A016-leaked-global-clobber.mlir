// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// A016: call to external function clobbers global.
// The external function doesn't take the global as arg, but it's external
// so the pass conservatively clobbers globals.
// NOTE: the existing test call-clobbers-global.mlir shows that calling an
// external func that does NOT take the memref as arg does NOT clobber it
// (result is SINGLE). So we pass the global to the external func.

module {
  memref.global "private" @g : memref<i32> = uninitialized

  func.func private @external_clobber(%m: memref<i32>)

  func.func @leaked_global_clobber() {
    %g = memref.get_global @g : memref<i32>
    %c42 = arith.constant 42 : i32
    memref.store %c42, %g[] : memref<i32>
    call @external_clobber(%g) : (memref<i32>) -> ()
    // expected-remark @below {{load: LEAKED}}
    %v = memref.load %g[] : memref<i32>
    return
  }
}
