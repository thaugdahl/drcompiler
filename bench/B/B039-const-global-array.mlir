// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// B039: Constant global fold — load from constant global f64 scalar.
// Tests constant-global-fold with a different element type (f64).

module {
  memref.global "private" constant @gf64 : memref<f64> = dense<3.14>

  func.func @const_global_f64() -> f64 {
    %g = memref.get_global @gf64 : memref<f64>
    // expected-remark @below {{constant-global-fold: ACCEPT}}
    %v = memref.load %g[] : memref<f64>
    return %v : f64
  }
}

// The load is folded to the constant.
// CHECK-LABEL: func.func @const_global_f64
// CHECK:         %[[C:.*]] = arith.constant 3.14{{0*}}e+00 : f64
// CHECK:         return %[[C]] : f64
