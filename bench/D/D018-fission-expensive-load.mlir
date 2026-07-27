// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission{test-diagnostics l1-latency=12})' -verify-diagnostics | FileCheck %s
//
// D018: simulate expensive buffer loads by overriding l1-latency=12.
// Chain: mulf(3) + addf(1) + divf(15) = 19.  consumers=2.
// keep = 19 + 1 + 2*12 = 44.  recompute = 2*19 = 38.
// 44 > 38 → SKIP (buffer load too expensive).

module {
  func.func @run(%x: memref<?xf64>, %n: index) -> f64 {
    %cst0 = arith.constant 0.0 : f64
    %cst1 = arith.constant 1.0 : f64

    // expected-remark @below {{memory-fission: SKIP (compute=19, consumers=2)}}
    %sum = affine.for %i = 0 to %n iter_args(%acc = %cst0) -> (f64) {
      %xi = affine.load %x[%i] : memref<?xf64>
      %sq = arith.mulf %xi, %xi : f64
      %s  = arith.addf %sq, %cst1 : f64
      %v  = arith.divf %s, %xi : f64
      %out = arith.addf %acc, %v : f64
      affine.yield %out : f64
    }

    %max = affine.for %i = 0 to %n iter_args(%best = %cst0) -> (f64) {
      %xi = affine.load %x[%i] : memref<?xf64>
      %sq = arith.mulf %xi, %xi : f64
      %s  = arith.addf %sq, %cst1 : f64
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
// No fission — SKIP, buffer load too expensive with l1-latency=12.
// CHECK:         affine.for
// CHECK:           arith.divf
// CHECK:         affine.for
// CHECK:           arith.divf
// CHECK-NOT:     memref.alloc
