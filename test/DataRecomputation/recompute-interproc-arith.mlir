// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute})' | FileCheck %s

// Callee computes arg0 + constant and stores through output arg.
// Caller's load replaced with rematerialized call_operand + constant.

module {
  func.func private @add_and_store(%x: i32, %out: memref<i32>) {
    %c10 = arith.constant 10 : i32
    %sum = arith.addi %x, %c10 : i32
    memref.store %sum, %out[] : memref<i32>
    return
  }

  func.func @caller(%input: i32) -> i32 {
    %alloc = memref.alloc() : memref<i32>
    call @add_and_store(%input, %alloc) : (i32, memref<i32>) -> ()
    %val = memref.load %alloc[] : memref<i32>
    memref.dealloc %alloc : memref<i32>
    return %val : i32
  }
}

// CHECK-LABEL: func.func @caller
// CHECK-SAME:    %[[INPUT:.*]]: i32
// CHECK:         call @add_and_store
// CHECK:         %[[C10:.*]] = arith.constant 10 : i32
// CHECK:         %[[SUM:.*]] = arith.addi %[[INPUT]], %[[C10]] : i32
// CHECK-NOT:     memref.load
// CHECK:         return %[[SUM]] : i32
