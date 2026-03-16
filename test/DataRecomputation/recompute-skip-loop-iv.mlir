// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute})' | FileCheck %s

// Stored value depends on loop IV that doesn't dominate the load: no transformation.

module {
  func.func @test() -> index {
    %alloc = memref.alloc() : memref<index>
    affine.for %i = 0 to 10 {
      memref.store %i, %alloc[] : memref<index>
    }
    %val = memref.load %alloc[] : memref<index>
    memref.dealloc %alloc : memref<index>
    return %val : index
  }
}

// CHECK-LABEL: func.func @test
// CHECK:         affine.for
// CHECK:           memref.store
// CHECK:         %[[VAL:.*]] = memref.load
// CHECK:         return %[[VAL]] : index
