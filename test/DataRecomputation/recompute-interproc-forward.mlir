// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute})' | FileCheck %s

// Callee stores a constant through output memref arg.
// Caller's load replaced with rematerialized constant.

module {
  func.func private @write_const(%out: memref<i32>) {
    %c99 = arith.constant 99 : i32
    memref.store %c99, %out[] : memref<i32>
    return
  }

  func.func @caller() -> i32 {
    %alloc = memref.alloc() : memref<i32>
    call @write_const(%alloc) : (memref<i32>) -> ()
    %val = memref.load %alloc[] : memref<i32>
    memref.dealloc %alloc : memref<i32>
    return %val : i32
  }
}

// CHECK-LABEL: func.func @caller
// CHECK:         call @write_const
// CHECK:         %[[C99:.*]] = arith.constant 99 : i32
// CHECK-NOT:     memref.load
// CHECK:         return %[[C99]] : i32
