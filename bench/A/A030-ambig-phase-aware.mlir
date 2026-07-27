// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// A030: store reachable only on 2nd iteration. First iteration reads before
// the store executes. Uses scf.for with a pre-loop store and in-loop store
// to create MULTI provenance.

module {
  func.func @ambig_phase_aware() {
    %alloc = memref.alloc() : memref<i32>
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %init = arith.constant 0 : i32
    %update = arith.constant 42 : i32

    // Pre-loop store sets initial value.
    memref.store %init, %alloc[] : memref<i32>

    scf.for %i = %c0 to %c10 step %c1 {
      // Read current value (on 1st iter: reads pre-loop store;
      // on 2nd+ iter: reads in-loop store from previous iter).
      // expected-remark @below {{load: MULTI}}
      %v = memref.load %alloc[] : memref<i32>
      // Update for next iteration.
      memref.store %update, %alloc[] : memref<i32>
    }

    // Post-loop load: reached by both pre-loop and in-loop stores.
    // expected-remark @below {{load: MULTI}}
    %final = memref.load %alloc[] : memref<i32>
    memref.dealloc %alloc : memref<i32>
    return
  }
}
