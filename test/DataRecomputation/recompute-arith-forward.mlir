// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute})' | FileCheck %s

// Arithmetic computation forwarded: store dominates load, load replaced.

module {
  func.func @test(%arg0: i32) -> i32 {
    %alloc = memref.alloc() : memref<i32>
    %c1 = arith.constant 1 : i32
    %sum = arith.addi %arg0, %c1 : i32
    memref.store %sum, %alloc[] : memref<i32>
    %val = memref.load %alloc[] : memref<i32>
    memref.dealloc %alloc : memref<i32>
    return %val : i32
  }
}

// CHECK-LABEL: func.func @test
// CHECK-SAME:    %[[ARG0:.*]]: i32
// CHECK:         %[[C1:.*]] = arith.constant 1 : i32
// CHECK:         %[[SUM:.*]] = arith.addi %[[ARG0]], %[[C1]] : i32
// CHECK-NOT:     memref.load
// CHECK:         return %[[SUM]] : i32
