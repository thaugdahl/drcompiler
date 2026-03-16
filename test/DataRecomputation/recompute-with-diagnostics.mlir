// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics dr-recompute})' -verify-diagnostics | FileCheck %s

// Both diagnostics and recompute: verify remarks AND load replaced.

module {
  func.func @test() -> i32 {
    %alloc = memref.alloc() : memref<i32>
    %c42 = arith.constant 42 : i32
    memref.store %c42, %alloc[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %val = memref.load %alloc[] : memref<i32>
    memref.dealloc %alloc : memref<i32>
    return %val : i32
  }
}

// CHECK-LABEL: func.func @test
// CHECK:         %[[C42:.*]] = arith.constant 42 : i32
// CHECK-NOT:     memref.load
// CHECK:         return %[[C42]] : i32
