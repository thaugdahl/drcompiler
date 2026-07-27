// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission{test-diagnostics})' -verify-diagnostics | FileCheck %s
//
// D003: exp+log chain duplicated in 2 loops → FISSION.
// Chain: exp(20) → log(20) chained.  exp is expensive (20>=15).
// exp has one user = log (cost 20 >= minConsumerCost=10), so tip = log.
// traceBack(log): log(20) + exp(20) + addf(1) + mulf(3) = 44.
// recompute = 2*44 = 88, keep = 44 + 1 + 2*4 = 53 → FISSION.

module {
  func.func @run(%x: memref<?xf64>, %n: index) -> f64 {
    %cst0 = arith.constant 0.0 : f64
    %cst1 = arith.constant 1.0 : f64

    // expected-remark @below {{memory-fission: FISSION (compute=44, consumers=2)}}
    // expected-remark @below {{materialized buffer for 2 consumers}}
    %sum = affine.for %i = 0 to %n iter_args(%acc = %cst0) -> (f64) {
      %xi = affine.load %x[%i] : memref<?xf64>
      %sq = arith.mulf %xi, %xi : f64
      %s  = arith.addf %sq, %cst1 : f64
      %e  = math.exp %s : f64
      %v  = math.log %e : f64
      %out = arith.addf %acc, %v : f64
      affine.yield %out : f64
    }

    %max = affine.for %i = 0 to %n iter_args(%best = %cst0) -> (f64) {
      %xi = affine.load %x[%i] : memref<?xf64>
      %sq = arith.mulf %xi, %xi : f64
      %s  = arith.addf %sq, %cst1 : f64
      %e  = math.exp %s : f64
      %v  = math.log %e : f64
      %gt = arith.cmpf ogt, %v, %best : f64
      %out = arith.select %gt, %v, %best : f64
      affine.yield %out : f64
    }

    %result = arith.addf %sum, %max : f64
    return %result : f64
  }
}

// CHECK-LABEL: func.func @run
// CHECK:         %[[BUF:.*]] = memref.alloc
// CHECK:         affine.for
// CHECK:           math.exp
// CHECK:           math.log
// CHECK:           affine.store %{{.*}}, %[[BUF]]
// CHECK:         affine.for
// CHECK:           affine.load %[[BUF]]
// CHECK-NOT:       math.exp
// CHECK:           affine.yield
// CHECK:         affine.for
// CHECK:           affine.load %[[BUF]]
// CHECK-NOT:       math.exp
// CHECK:           affine.yield
// CHECK:         memref.dealloc %[[BUF]]
