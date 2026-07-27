// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// B010: Full remat — chain where 2/3 ops are constants (zero cost).

module {
  func.func @remat_with_constant() -> i32 {
    %c0 = arith.constant 0 : index

    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %buf = memref.alloc() : memref<1xi32>

    %c10 = arith.constant 10 : i32
    %c20 = arith.constant 20 : i32
    %val = arith.addi %c10, %c20 : i32
    memref.store %val, %buf[%c0] : memref<1xi32>

    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %buf[%c0] : memref<1xi32>
    memref.dealloc %buf : memref<1xi32>
    return %v : i32
  }
}

// CHECK-LABEL: func.func @remat_with_constant
// CHECK-NOT:     memref.load
// CHECK:         return
