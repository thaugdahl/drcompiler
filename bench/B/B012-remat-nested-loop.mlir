// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// B012: Full remat inside doubly-nested affine loop.

module {
  func.func @remat_nested_loop(%x: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : i32
    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %buf = memref.alloc() : memref<i32>
    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %out = memref.alloc() : memref<4x4xi32>
    %val = arith.addi %x, %c1 : i32
    memref.store %val, %buf[] : memref<i32>

    affine.for %i = 0 to 4 {
      affine.for %j = 0 to 4 {
        // expected-remark @below {{direct-forward: ACCEPT}}
        // expected-remark @below {{load: SINGLE}}
        %v = memref.load %buf[] : memref<i32>
        affine.store %v, %out[%i, %j] : memref<4x4xi32>
      }
    }

    // expected-remark @below {{load: SINGLE}}
    // expected-remark @below {{full-remat: ACCEPT}}
    %r = affine.load %out[%c0, %c0] : memref<4x4xi32>
    memref.dealloc %buf : memref<i32>
    memref.dealloc %out : memref<4x4xi32>
    return %r : i32
  }
}

// CHECK-LABEL: func.func @remat_nested_loop
// CHECK-NOT:     memref.load %{{.*}}[] : memref<i32>
// CHECK:         return
