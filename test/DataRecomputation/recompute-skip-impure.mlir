// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute})' | FileCheck %s

// Stored value depends on a memref.load (impure) inside a loop.
// Store doesn't dominate load, and rematerialization is rejected
// because the operand tree contains a memory-effectful op.

module {
  func.func @test(%src: memref<i32>) -> i32 {
    %alloc = memref.alloc() : memref<i32>
    affine.for %i = 0 to 10 {
      %v = memref.load %src[] : memref<i32>
      memref.store %v, %alloc[] : memref<i32>
    }
    %val = memref.load %alloc[] : memref<i32>
    memref.dealloc %alloc : memref<i32>
    return %val : i32
  }
}

// CHECK-LABEL: func.func @test
// CHECK:         affine.for
// CHECK:           memref.load
// CHECK:           memref.store
// CHECK:         %[[VAL:.*]] = memref.load
// CHECK:         return %[[VAL]] : i32
