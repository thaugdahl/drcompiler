// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// B040: Constant global fold — same constant global loaded in 4 functions.

module {
  memref.global "private" constant @shared : memref<i32> = dense<99>

  func.func @user1() -> i32 {
    %g = memref.get_global @shared : memref<i32>
    // expected-remark @below {{constant-global-fold: ACCEPT}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func @user2() -> i32 {
    %g = memref.get_global @shared : memref<i32>
    // expected-remark @below {{constant-global-fold: ACCEPT}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func @user3() -> i32 {
    %g = memref.get_global @shared : memref<i32>
    // expected-remark @below {{constant-global-fold: ACCEPT}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func @user4() -> i32 {
    %g = memref.get_global @shared : memref<i32>
    // expected-remark @below {{constant-global-fold: ACCEPT}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }
}

// All loads folded to the constant.
// CHECK-LABEL: func.func @user1
// CHECK:         %[[C:.*]] = arith.constant 99 : i32
// CHECK:         return %[[C]] : i32
// CHECK-LABEL: func.func @user2
// CHECK:         %[[C:.*]] = arith.constant 99 : i32
// CHECK:         return %[[C]] : i32
// CHECK-LABEL: func.func @user3
// CHECK:         %[[C:.*]] = arith.constant 99 : i32
// CHECK:         return %[[C]] : i32
// CHECK-LABEL: func.func @user4
// CHECK:         %[[C:.*]] = arith.constant 99 : i32
// CHECK:         return %[[C]] : i32
