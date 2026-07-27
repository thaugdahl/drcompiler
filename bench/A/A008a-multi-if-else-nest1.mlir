// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// A008a: single if/else with different stores -> MULTI.

module {
  func.func @multi_if_else_nest1(%cond: i1) {
    %alloc = memref.alloc() : memref<f32>
    %a = arith.constant 1.0 : f32
    %b = arith.constant 2.0 : f32
    scf.if %cond {
      memref.store %a, %alloc[] : memref<f32>
    } else {
      memref.store %b, %alloc[] : memref<f32>
    }
    // expected-remark @below {{load: MULTI}}
    %v = memref.load %alloc[] : memref<f32>
    memref.dealloc %alloc : memref<f32>
    return
  }
}
