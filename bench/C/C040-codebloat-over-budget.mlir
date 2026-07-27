// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-buffer-elim dr-test-diagnostics dr-icache-soft-budget=4})' -verify-diagnostics

// C040: Code bloat — over budget.
// 1 distinct compute with tree size 6 → bloatOps = 1*6 = 6 > 4.
// codeBloatPenalty = 6 - 4 = 2.  With alloca (no alloc overhead):
// keep = 1*4 + 1*2 = 6, elim = perElemCost(6) + bloat(2) = 8.
// 8 > 6 → INFEASIBLE.

module {
  func.func @codebloat_over(%x: f64) -> f64 {
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
