// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// B038: Constant global fold — load from constant global scalar.

module {
  memref.global "private" constant @gscalar : memref<i32> = dense<42>

  func.func @const_global_scalar() -> i32 {
    %g = memref.get_global @gscalar : memref<i32>
    // expected-remark @below {{constant-global-fold: ACCEPT}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }
}

// The load is folded to a constant.
// CHECK-LABEL: func.func @const_global_scalar
// CHECK:         %[[C:.*]] = arith.constant 42 : i32
// CHECK:         return %[[C]] : i32
