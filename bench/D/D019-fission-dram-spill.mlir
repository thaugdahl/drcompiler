// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission{test-diagnostics l1-latency=200})' -verify-diagnostics | FileCheck %s
//
// D019: simulate DRAM-level buffer cost by overriding l1-latency=200.
// Chain: mulf(3) + addf(1) + sqrt(20) + addf(1) + divf(15) = 40.  consumers=2.
// keep = 40 + 1 + 2*200 = 441.  recompute = 2*40 = 80.
// 441 > 80 → SKIP (buffer load extremely expensive at DRAM latency).

module {
  func.func @run(%x: memref<?xf64>, %n: index) -> f64 {
    %cst0 = arith.constant 0.0 : f64
    %cst1 = arith.constant 1.0 : f64
    %eps  = arith.constant 0.001 : f64

    // expected-remark @below {{memory-fission: SKIP (compute=40, consumers=2)}}
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
// No fission — SKIP, DRAM-level load latency makes buffer too expensive.
// CHECK:         affine.for
// CHECK:           math.sqrt
// CHECK:           arith.divf
// CHECK:         affine.for
// CHECK:           math.sqrt
// CHECK:           arith.divf
// CHECK-NOT:     memref.alloc
