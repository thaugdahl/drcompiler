// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute})' | FileCheck %s

// Negative case: inner load has MULTI provenance (two possible stores).
// Chaining must be rejected; the final load should remain.

module {
  func.func @test(%cond: i1) -> i32 {
    %a = memref.alloc() : memref<i32>
    %b = memref.alloc() : memref<i32>
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    scf.if %cond {
      memref.store %c1, %a[] : memref<i32>
    } else {
      memref.store %c2, %a[] : memref<i32>
    }
    // %a has MULTI provenance (two stores from if/else).
    affine.for %i = 0 to 10 {
      %v = memref.load %a[] : memref<i32>
      %sum = arith.addi %v, %c1 : i32
      memref.store %sum, %b[] : memref<i32>
    }
    %result = memref.load %b[] : memref<i32>
    memref.dealloc %a : memref<i32>
    memref.dealloc %b : memref<i32>
    return %result : i32
  }
}

// CHECK-LABEL: func.func @test
// CHECK:         scf.if
// CHECK:         affine.for
// CHECK:           memref.load
// CHECK:           memref.store
// The final load should NOT be eliminated:
// CHECK:         %[[VAL:.*]] = memref.load
// CHECK:         return %[[VAL]] : i32
