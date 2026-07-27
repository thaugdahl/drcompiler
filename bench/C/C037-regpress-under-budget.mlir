// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-buffer-elim dr-test-diagnostics dr-reg-budget-fp=32 dr-spill-reload=4})' -verify-diagnostics

// C037: Register pressure — under budget (RegisterPressureAnalysis path).
// Stored-value addf chain has ~5 FP values live at peak; fp_budget=32
// leaves the analysis at zero spill cycles -> no register-pressure penalty.
// buffer-elim should be FEASIBLE.

module {
  func.func @regpress_under(%x: f64) -> f64 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f64

    // expected-remark @below {{cost-model: RECOMPUTE}}
    // expected-remark @below {{buffer-elim: FEASIBLE}}
    %buf = memref.alloc() : memref<1xf64>

    %a1 = arith.addf %x, %one : f64
    %a2 = arith.addf %a1, %one : f64
    %a3 = arith.addf %a2, %one : f64
    %val = arith.addf %a3, %one : f64
    memref.store %val, %buf[%c0] : memref<1xf64>

    // expected-remark @below {{load: SINGLE}}
    // expected-remark @below {{direct-forward: ACCEPT}}
    %v = memref.load %buf[%c0] : memref<1xf64>

    memref.dealloc %buf : memref<1xf64>
    return %v : f64
  }
}
