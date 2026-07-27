// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-buffer-elim dr-test-diagnostics dr-reg-budget-fp=2 dr-spill-reload=20})' -verify-diagnostics

// C038: Register pressure — over budget (RegisterPressureAnalysis path).
// fp_budget=2 forces non-trivial sum-excess across the addf chain; combined
// with spill_reload=20 the regPressurePenalty pushes elim cost above keep.

module {
  func.func @regpress_over(%x: f64) -> f64 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f64

    // expected-remark @below {{cost-model: RECOMPUTE}}
    // expected-remark @below {{buffer-elim:}}
    %buf = memref.alloca() : memref<1xf64>

    %a1 = arith.addf %x, %one : f64
    %a2 = arith.addf %a1, %one : f64
    %a3 = arith.addf %a2, %one : f64
    %a4 = arith.addf %a3, %one : f64
    %a5 = arith.addf %a4, %one : f64
    %val = arith.addf %a5, %one : f64
    memref.store %val, %buf[%c0] : memref<1xf64>

    // expected-remark @below {{load: SINGLE}}
    // expected-remark @below {{direct-forward: ACCEPT}}
    %v = memref.load %buf[%c0] : memref<1xf64>

    return %v : f64
  }
}
