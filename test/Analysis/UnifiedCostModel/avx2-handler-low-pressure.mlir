// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-buffer-elim dr-test-diagnostics dr-arch-handler=x86-64-avx2})' -verify-diagnostics

// Phase 3 sanity: pick x86-64-avx2 handler explicitly.  Three f64 SSA
// values live at the addf chain peak — well under fp_budget=16 — so the
// unified cost model reports zero register-pressure penalty and buffer-elim
// is FEASIBLE.

module {
  func.func @fp_under(%x: f64) -> f64 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f64

    // expected-remark @below {{cost-model: RECOMPUTE}}
    // expected-remark @below {{buffer-elim: FEASIBLE}}
    %buf = memref.alloc() : memref<1xf64>

    %v1 = arith.addf %x, %one : f64
    %v2 = arith.addf %v1, %one : f64
    %v3 = arith.addf %v2, %one : f64
    memref.store %v3, %buf[%c0] : memref<1xf64>

    // expected-remark @below {{load: SINGLE}}
    // expected-remark @below {{direct-forward: ACCEPT}}
    %v = memref.load %buf[%c0] : memref<1xf64>

    memref.dealloc %buf : memref<1xf64>
    return %v : f64
  }
}
