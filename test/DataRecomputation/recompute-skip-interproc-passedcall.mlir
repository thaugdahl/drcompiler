// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute})' | FileCheck %s

// Callee passes memref to further call: provenance is LEAKED, not SINGLE.
// No transformation should happen.

module {
  func.func private @inner(%m: memref<i32>)

  func.func private @outer(%out: memref<i32>) {
    %c1 = arith.constant 1 : i32
    memref.store %c1, %out[] : memref<i32>
    func.call @inner(%out) : (memref<i32>) -> ()
    return
  }

  func.func @caller() -> i32 {
    %alloc = memref.alloc() : memref<i32>
    call @outer(%alloc) : (memref<i32>) -> ()
    %val = memref.load %alloc[] : memref<i32>
    memref.dealloc %alloc : memref<i32>
    return %val : i32
  }
}

// CHECK-LABEL: func.func @caller
// CHECK:         call @outer
// CHECK:         %[[VAL:.*]] = memref.load
// CHECK:         return %[[VAL]] : i32
