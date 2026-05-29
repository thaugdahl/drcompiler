// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-buffer-elim dr-test-diagnostics dr-arch-handler=x86-64-avx2 dr-reg-budget-fp=2 dr-spill-reload=20})' -verify-diagnostics

// Force fp_budget=2 on AVX2 to overflow on the same chain.  cost-model
// still RECOMPUTEs (tiny ALU cost), but buffer-elim reflects the unified
// register-pressure penalty in its keep/elim numbers.

module {
  func.func @fp_tight(%x: f64) -> f64 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f64

    // expected-remark @below {{cost-model: RECOMPUTE}}
    // expected-remark @below {{buffer-elim:}}
    %buf = memref.alloca() : memref<1xf64>

    %v1 = arith.addf %x, %one : f64
    %v2 = arith.addf %v1, %one : f64
    %v3 = arith.addf %v2, %one : f64
    %v4 = arith.addf %v3, %one : f64
    %v5 = arith.addf %v4, %one : f64
    memref.store %v5, %buf[%c0] : memref<1xf64>

    // expected-remark @below {{load: SINGLE}}
    // expected-remark @below {{direct-forward: ACCEPT}}
    %v = memref.load %buf[%c0] : memref<1xf64>

    return %v : f64
  }
}
