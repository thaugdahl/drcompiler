// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// S1-A: scf.for with dynamic bounds (may execute zero times) and no prior
// store. Post-loop load must NOT be SINGLE.

module {
  func.func @for_may_zero(%lb: index, %ub: index, %step: index) -> i32 {
    %alloc = memref.alloc() : memref<i32>
    %c42 = arith.constant 42 : i32
    scf.for %i = %lb to %ub step %step {
      memref.store %c42, %alloc[] : memref<i32>
    }
    // expected-remark @below {{load: LEAKED}}
    %v = memref.load %alloc[] : memref<i32>
    memref.dealloc %alloc : memref<i32>
    return %v : i32
  }
}
