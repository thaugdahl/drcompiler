// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// B011: Full remat — chain with sitofp/fptosi type conversions.

module {
  func.func @remat_type_convert(%x: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f32

    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %buf = memref.alloc() : memref<1xi32>

    %f = arith.sitofp %x : i32 to f32
    %added = arith.addf %f, %one : f32
    %back = arith.fptosi %added : f32 to i32
    memref.store %back, %buf[%c0] : memref<1xi32>

    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %buf[%c0] : memref<1xi32>
    memref.dealloc %buf : memref<1xi32>
    return %v : i32
  }
}

// CHECK-LABEL: func.func @remat_type_convert
// CHECK-NOT:     memref.load
// CHECK:         return
