// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute})' | FileCheck %s

// Multi-op pure chain rematerialized: store inside affine.for with
// provably-positive trip count. The chain arg0 + 1 is rematerialized
// at the load point since all leaves (arg0, constant) dominate the load.

module {
  func.func @test(%arg0: i32) -> i32 {
    %alloc = memref.alloc() : memref<i32>
    %c1 = arith.constant 1 : i32
    affine.for %i = 0 to 10 {
      %sum = arith.addi %arg0, %c1 : i32
      memref.store %sum, %alloc[] : memref<i32>
    }
    %val = memref.load %alloc[] : memref<i32>
    memref.dealloc %alloc : memref<i32>
    return %val : i32
  }
}

// CHECK-LABEL: func.func @test
// CHECK-SAME:    %[[ARG0:.*]]: i32
// CHECK:         affine.for
// CHECK:         %[[C1:.*]] = arith.constant 1 : i32
// CHECK:         %[[SUM:.*]] = arith.addi %[[ARG0]], %[[C1]] : i32
// CHECK-NOT:     memref.load
// CHECK:         return %[[SUM]] : i32
