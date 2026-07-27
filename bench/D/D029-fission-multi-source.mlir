// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission{test-diagnostics})' -verify-diagnostics | FileCheck %s
//
// D029: chain uses a second memref for a loop-invariant value loaded before
// the loops.  The chain's sourceMemref is %x (the indexed load inside the
// loop); the invariant load from %y is hoisted outside.
// Chain: mulf(3) + addf(1) + divf(15) = 19.  consumers=2.
// recompute = 2*19 = 38, keep = 19 + 1 + 2*4 = 28 → FISSION.

module {
  func.func @run(%x: memref<?xf64>, %scale: f64, %n: index) -> f64 {
    %cst0 = arith.constant 0.0 : f64

    // expected-remark @below {{memory-fission: FISSION (compute=19, consumers=2)}}
    // expected-remark @below {{materialized buffer for 2 consumers}}
    %sum = affine.for %i = 0 to %n iter_args(%acc = %cst0) -> (f64) {
      %xi = affine.load %x[%i] : memref<?xf64>
      %sq = arith.mulf %xi, %xi : f64
      %s  = arith.addf %sq, %scale : f64
      %v  = arith.divf %s, %xi : f64
      %out = arith.addf %acc, %v : f64
      affine.yield %out : f64
    }

    %max = affine.for %i = 0 to %n iter_args(%best = %cst0) -> (f64) {
      %xi = affine.load %x[%i] : memref<?xf64>
      %sq = arith.mulf %xi, %xi : f64
      %s  = arith.addf %sq, %scale : f64
      %v  = arith.divf %s, %xi : f64
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
// CHECK:           arith.divf
// CHECK:           affine.store %{{.*}}, %[[BUF]]
// CHECK:         affine.for
// CHECK:           affine.load %[[BUF]]
// CHECK-NOT:       arith.divf
// CHECK:           affine.yield
// CHECK:         affine.for
// CHECK:           affine.load %[[BUF]]
// CHECK-NOT:       arith.divf
// CHECK:           affine.yield
// CHECK:         memref.dealloc %[[BUF]]
