// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// B004: Direct forward — store loop-invariant before loop, load inside loop.

module {
  func.func @direct_loop_invariant(%x: i32) -> i32 {
    %c0 = arith.constant 0 : index
    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %buf = memref.alloc() : memref<i32>
    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %out = memref.alloc() : memref<10xi32>
    %c1 = arith.constant 1 : i32
    %val = arith.addi %x, %c1 : i32
    memref.store %val, %buf[] : memref<i32>

    affine.for %i = 0 to 10 {
      // expected-remark @below {{direct-forward: ACCEPT}}
      // expected-remark @below {{load: SINGLE}}
      %v = memref.load %buf[] : memref<i32>
      affine.store %v, %out[%i] : memref<10xi32>
    }

    // expected-remark @below {{load: SINGLE}}
    // expected-remark @below {{full-remat: ACCEPT}}
    %r = affine.load %out[%c0] : memref<10xi32>
    memref.dealloc %buf : memref<i32>
    memref.dealloc %out : memref<10xi32>
    return %r : i32
  }
}

// CHECK-LABEL: func.func @direct_loop_invariant
// CHECK-NOT:     memref.load %{{.*}}[] : memref<i32>
// CHECK:         return
