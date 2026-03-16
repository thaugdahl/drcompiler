// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission{test-diagnostics})' -verify-diagnostics | FileCheck %s

// Two loops with identical expensive computation (sqrt + div, cost=40) from
// the same source memref.  Cost model: recompute=80 vs keep=49 → FISSION.
// The pass should create a producer loop + buffer and replace the computation
// in both consumer loops with buffer loads.

module {
  func.func @run(%x: memref<?xf64>, %n: index) -> f64 {
    %cst0 = arith.constant 0.0 : f64
    %cst1 = arith.constant 1.0 : f64
    %eps  = arith.constant 0.001 : f64

    // expected-remark @below {{memory-fission: FISSION}}
    // expected-remark @below {{materialized buffer for 2 consumers}}
    %sum = affine.for %i = 0 to %n iter_args(%acc = %cst0) -> (f64) {
      %xi = affine.load %x[%i] : memref<?xf64>
      %sq = arith.mulf %xi, %xi : f64
      %s  = arith.addf %sq, %cst1 : f64
      %r  = math.sqrt %s : f64
      %d  = arith.addf %xi, %eps : f64
      %v  = arith.divf %r, %d : f64
      %out = arith.addf %acc, %v : f64
      affine.yield %out : f64
    }

    %max = affine.for %i = 0 to %n iter_args(%best = %cst0) -> (f64) {
      %xi = affine.load %x[%i] : memref<?xf64>
      %sq = arith.mulf %xi, %xi : f64
      %s  = arith.addf %sq, %cst1 : f64
      %r  = math.sqrt %s : f64
      %d  = arith.addf %xi, %eps : f64
      %v  = arith.divf %r, %d : f64
      %gt = arith.cmpf ogt, %v, %best : f64
      %out = arith.select %gt, %v, %best : f64
      affine.yield %out : f64
    }

    %result = arith.addf %sum, %max : f64
    return %result : f64
  }
}

// CHECK-LABEL: func.func @run
// Producer loop: computes and stores into buffer.
// CHECK:         %[[BUF:.*]] = memref.alloc
// CHECK:         affine.for
// CHECK:           math.sqrt
// CHECK:           arith.divf
// CHECK:           affine.store %{{.*}}, %[[BUF]]
// Consumer 1: loads from buffer (no sqrt/div).
// CHECK:         affine.for
// CHECK:           affine.load %[[BUF]]
// CHECK-NOT:       math.sqrt
// CHECK:           affine.yield
// Consumer 2: loads from buffer (no sqrt/div).
// CHECK:         affine.for
// CHECK:           affine.load %[[BUF]]
// CHECK-NOT:       math.sqrt
// CHECK:           affine.yield
// CHECK:         memref.dealloc %[[BUF]]
