// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute})' | FileCheck %s

// Constant rematerialized from non-dominating region (then-branch of if).
// The affine.for with provably-positive trip count (lb=0, ub=10) makes
// the store SINGLE, and the constant can be rematerialized at the load point.

module {
  func.func @test() -> i32 {
    %alloc = memref.alloc() : memref<i32>
    %c42 = arith.constant 42 : i32
    affine.for %i = 0 to 10 {
      memref.store %c42, %alloc[] : memref<i32>
    }
    %val = memref.load %alloc[] : memref<i32>
    memref.dealloc %alloc : memref<i32>
    return %val : i32
  }
}

// CHECK-LABEL: func.func @test
// CHECK:         affine.for
// CHECK:         %[[C42:.*]] = arith.constant 42 : i32
// CHECK-NOT:     memref.load
// CHECK:         return %[[C42]] : i32
