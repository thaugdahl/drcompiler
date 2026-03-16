// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute})' | FileCheck %s

// Single-level load chain: the stored value's operand tree contains a
// memref.load whose provenance is SINGLE. The pass chains through it
// to find a pure computation (arith.addi of a constant and a func arg).

module {
  func.func @test(%arg0: i32) -> i32 {
    %a = memref.alloc() : memref<i32>
    %b = memref.alloc() : memref<i32>
    %c1 = arith.constant 1 : i32
    memref.store %c1, %a[] : memref<i32>
    affine.for %i = 0 to 10 {
      %v = memref.load %a[] : memref<i32>
      %sum = arith.addi %v, %arg0 : i32
      memref.store %sum, %b[] : memref<i32>
    }
    %result = memref.load %b[] : memref<i32>
    memref.dealloc %a : memref<i32>
    memref.dealloc %b : memref<i32>
    return %result : i32
  }
}

// CHECK-LABEL: func.func @test
// CHECK-SAME:    %[[ARG0:.*]]: i32
// CHECK:         affine.for
// CHECK:         %[[C1:.*]] = arith.constant 1 : i32
// CHECK:         %[[SUM:.*]] = arith.addi %[[C1]], %[[ARG0]] : i32
// CHECK-NOT:     memref.load %{{.*}}[] : memref<i32>
// The final load of %b should be replaced:
// CHECK:         return %[[SUM]] : i32
