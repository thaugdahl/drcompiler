// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute})' | FileCheck %s

// Multi-level load chain: load→store→load→store→load.
// %a stores a constant, %b loads from %a and adds 1, %c loads from %b
// and adds 1. The final load of %c should chain through both loads
// to rematerialize the pure computation.

module {
  func.func @test() -> i32 {
    %a = memref.alloc() : memref<i32>
    %b = memref.alloc() : memref<i32>
    %c = memref.alloc() : memref<i32>
    %c10 = arith.constant 10 : i32
    %c1 = arith.constant 1 : i32
    memref.store %c10, %a[] : memref<i32>
    affine.for %i = 0 to 5 {
      %va = memref.load %a[] : memref<i32>
      %sum1 = arith.addi %va, %c1 : i32
      memref.store %sum1, %b[] : memref<i32>
    }
    affine.for %j = 0 to 5 {
      %vb = memref.load %b[] : memref<i32>
      %sum2 = arith.addi %vb, %c1 : i32
      memref.store %sum2, %c[] : memref<i32>
    }
    %result = memref.load %c[] : memref<i32>
    memref.dealloc %a : memref<i32>
    memref.dealloc %b : memref<i32>
    memref.dealloc %c : memref<i32>
    return %result : i32
  }
}

// The final load should be rematerialized as a chain of adds on constants.
// CHECK-LABEL: func.func @test
// CHECK:         affine.for
// CHECK:         affine.for
// CHECK:         }
// After both loops, the rematerialized chain appears:
// CHECK:         %[[C10:.*]] = arith.constant 10 : i32
// CHECK:         %[[S1:.*]] = arith.addi %[[C10]], %{{.*}} : i32
// CHECK:         %[[S2:.*]] = arith.addi %[[S1]], %{{.*}} : i32
// CHECK-NOT:     memref.load
// CHECK:         return %[[S2]] : i32
