// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-buffer-elim dr-test-diagnostics dr-spill-strategy=graph-color dr-reg-budget-fp=2 dr-spill-reload=10})' -verify-diagnostics

// Phase 3 sanity: graph-color spill strategy selected via CLI.  Should not
// crash and still produce a buffer-elim verdict; behavioural numbers
// differ from excess-hot but the structure of the decision pipeline is
// the same.

module {
  func.func @graph_color_ok(%x: f64) -> f64 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f64

    // expected-remark @below {{cost-model: RECOMPUTE}}
    // expected-remark @below {{buffer-elim:}}
    %buf = memref.alloca() : memref<1xf64>

    %a1 = arith.addf %x, %one : f64
    %a2 = arith.addf %a1, %one : f64
    %a3 = arith.addf %a2, %one : f64
    %val = arith.addf %a3, %one : f64
    memref.store %val, %buf[%c0] : memref<1xf64>

    // expected-remark @below {{load: SINGLE}}
    // expected-remark @below {{direct-forward: ACCEPT}}
    %v = memref.load %buf[%c0] : memref<1xf64>

    return %v : f64
  }
}
