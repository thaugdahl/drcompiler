// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// B032: Interproc — callee's stored value depends on a load from an arg memref.
// The callee is impure: rematerialization at caller is not safe.

module {
  func.func private @impure_write(%src: memref<i32>, %out: memref<i32>) {
    %v = memref.load %src[] : memref<i32>
    memref.store %v, %out[] : memref<i32>
    return
  }

  func.func @caller(%src: memref<i32>) -> i32 {
    // expected-remark @+1 {{cost-model:}}
    %alloc = memref.alloc() : memref<i32>
    call @impure_write(%src, %alloc) : (memref<i32>, memref<i32>) -> ()
    // expected-remark @below {{load: SINGLE}}
    // expected-remark @below {{interproc-remat: REJECT_UNSAFE}}
    // expected-remark @below {{interproc-cross: REJECT_PLAN}}
    %val = memref.load %alloc[] : memref<i32>
    memref.dealloc %alloc : memref<i32>
    return %val : i32
  }
}

// The load survives — impure callee blocks rematerialization.
// CHECK-LABEL: func.func @caller
// CHECK:         call @impure_write
// CHECK:         %[[VAL:.*]] = memref.load
// CHECK:         return %[[VAL]] : i32
