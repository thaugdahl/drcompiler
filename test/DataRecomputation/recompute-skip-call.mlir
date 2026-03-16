// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute})' | FileCheck %s

// Stored value comes from a call inside a loop.
// Store doesn't dominate load, and rematerialization is rejected
// because the operand tree contains a CallOpInterface.

module {
  func.func private @compute() -> i32

  func.func @test() -> i32 {
    %alloc = memref.alloc() : memref<i32>
    affine.for %i = 0 to 10 {
      %v = func.call @compute() : () -> i32
      memref.store %v, %alloc[] : memref<i32>
    }
    %val = memref.load %alloc[] : memref<i32>
    memref.dealloc %alloc : memref<i32>
    return %val : i32
  }
}

// CHECK-LABEL: func.func @test
// CHECK:         affine.for
// CHECK:           call @compute
// CHECK:           memref.store
// CHECK:         %[[VAL:.*]] = memref.load
// CHECK:         return %[[VAL]] : i32
