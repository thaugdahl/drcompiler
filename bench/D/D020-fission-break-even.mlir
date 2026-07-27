// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission{test-diagnostics l1-latency=7})' -verify-diagnostics | FileCheck %s
//
// D020: near break-even with l1-latency=7.
// Chain: divf(15) only (single expensive op).  consumers=2.
// keep = 15 + 1 + 2*7 = 30.  recompute = 2*15 = 30.
// 30 < 30 is false → SKIP (keep == recompute, not strictly less).

module {
  func.func @run(%x: memref<?xf64>, %n: index) -> f64 {
    %cst0 = arith.constant 0.0 : f64
    %cst1 = arith.constant 1.0 : f64

    // expected-remark @below {{memory-fission: SKIP (compute=15, consumers=2)}}
    %sum = affine.for %i = 0 to %n iter_args(%acc = %cst0) -> (f64) {
      %xi = affine.load %x[%i] : memref<?xf64>
      %v  = arith.divf %xi, %cst1 : f64
      %out = arith.addf %acc, %v : f64
      affine.yield %out : f64
    }

    %max = affine.for %i = 0 to %n iter_args(%best = %cst0) -> (f64) {
      %xi = affine.load %x[%i] : memref<?xf64>
      %v  = arith.divf %xi, %cst1 : f64
      %gt = arith.cmpf ogt, %v, %best : f64
      %out = arith.select %gt, %v, %best : f64
      affine.yield %out : f64
    }

    %result = arith.addf %sum, %max : f64
    return %result : f64
  }
}

// CHECK-LABEL: func.func @run
// No fission — exactly at break-even (not strictly less).
// CHECK:         affine.for
// CHECK:           arith.divf
// CHECK:         affine.for
// CHECK:           arith.divf
// CHECK-NOT:     memref.alloc
