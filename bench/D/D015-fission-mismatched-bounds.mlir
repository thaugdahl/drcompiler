// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission{test-diagnostics})' -verify-diagnostics | FileCheck %s
//
// D015: consumer loops with different upper bounds → NOT fission.
// Chain: mulf(3) + addf(1) + sqrt(20) = 24.  Same fingerprint, but first
// loop goes 0..%n while second goes 0..%m.  boundsMatch check fails.
// The FISSION remark is emitted (cost model says yes), but no transform.

module {
  func.func @run(%x: memref<?xf64>, %n: index, %m: index) -> f64 {
    %cst0 = arith.constant 0.0 : f64
    %cst1 = arith.constant 1.0 : f64

    // expected-remark @below {{memory-fission: FISSION (compute=24, consumers=2)}}
    %sum = affine.for %i = 0 to %n iter_args(%acc = %cst0) -> (f64) {
      %xi = affine.load %x[%i] : memref<?xf64>
      %sq = arith.mulf %xi, %xi : f64
      %s  = arith.addf %sq, %cst1 : f64
      %r  = math.sqrt %s : f64
      %out = arith.addf %acc, %r : f64
      affine.yield %out : f64
    }

    %max = affine.for %i = 0 to %m iter_args(%best = %cst0) -> (f64) {
      %xi = affine.load %x[%i] : memref<?xf64>
      %sq = arith.mulf %xi, %xi : f64
      %s  = arith.addf %sq, %cst1 : f64
      %r  = math.sqrt %s : f64
      %gt = arith.cmpf ogt, %r, %best : f64
      %out = arith.select %gt, %r, %best : f64
      affine.yield %out : f64
    }

    %result = arith.addf %sum, %max : f64
    return %result : f64
  }
}

// CHECK-LABEL: func.func @run
// No fission — bounds differ.
// CHECK-NOT:     memref.alloc
// CHECK:         affine.for
// CHECK:           math.sqrt
// CHECK:         affine.for
// CHECK:           math.sqrt
