// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute})' | FileCheck %s

// Callee's stored value depends on a load (impure computation): no transformation.

module {
  func.func private @impure_write(%src: memref<i32>, %out: memref<i32>) {
    %v = memref.load %src[] : memref<i32>
    memref.store %v, %out[] : memref<i32>
    return
  }

  func.func @caller(%src: memref<i32>) -> i32 {
    %alloc = memref.alloc() : memref<i32>
    call @impure_write(%src, %alloc) : (memref<i32>, memref<i32>) -> ()
    %val = memref.load %alloc[] : memref<i32>
    memref.dealloc %alloc : memref<i32>
    return %val : i32
  }
}

// CHECK-LABEL: func.func @caller
// CHECK:         call @impure_write
// CHECK:         %[[VAL:.*]] = memref.load
// CHECK:         return %[[VAL]] : i32
