// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission{test-diagnostics})' -verify-diagnostics | FileCheck %s
//
// D007: expensive chain in only 1 loop (no duplication) → NOT fission.
// Chain: mulf(3) + addf(1) + sqrt(20) = 24, but only 1 consumer.
// No candidate formed because indices.size() < 2.

module {
  func.func @run(%x: memref<?xf64>, %n: index) -> f64 {
    %cst0 = arith.constant 0.0 : f64
    %cst1 = arith.constant 1.0 : f64

    %sum = affine.for %i = 0 to %n iter_args(%acc = %cst0) -> (f64) {
      %xi = affine.load %x[%i] : memref<?xf64>
      %sq = arith.mulf %xi, %xi : f64
      %s  = arith.addf %sq, %cst1 : f64
      %r  = math.sqrt %s : f64
      %out = arith.addf %acc, %r : f64
      affine.yield %out : f64
    }

    return %sum : f64
  }
}

// CHECK-LABEL: func.func @run
// No fission — only one consumer loop.
// CHECK:         affine.for
// CHECK:           math.sqrt
// CHECK:           affine.yield
// CHECK-NOT:     memref.alloc
