// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission{test-diagnostics cpu-cost-model-file=%S/custom-costs.json})' -verify-diagnostics | FileCheck %s
//
// D021: uses cpu-cost-model-file with custom op costs.
// JSON: sqrt=40, divf=30, mulf=3, addf=1, constant=0.
// Chain (sqrt→divf tip extension): divf(30) + sqrt(40) + addf(1) + mulf(3)
//   + constant(0) + addf(1) + constant(0) = 75.
// recompute = 2*75 = 150, keep = 75 + 1 + 2*4 = 84 → FISSION.

module {
  func.func @run(%x: memref<?xf64>, %n: index) -> f64 {
    %cst0 = arith.constant 0.0 : f64
    %cst1 = arith.constant 1.0 : f64
    %eps  = arith.constant 0.001 : f64

    // expected-remark @below {{memory-fission: FISSION (compute=75, consumers=2)}}
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
// CHECK:         %[[BUF:.*]] = memref.alloc
// CHECK:         affine.for
// CHECK:           math.sqrt
// CHECK:           arith.divf
// CHECK:           affine.store %{{.*}}, %[[BUF]]
// CHECK:         affine.for
// CHECK:           affine.load %[[BUF]]
// CHECK-NOT:       math.sqrt
// CHECK:           affine.yield
// CHECK:         affine.for
// CHECK:           affine.load %[[BUF]]
// CHECK-NOT:       math.sqrt
// CHECK:           affine.yield
// CHECK:         memref.dealloc %[[BUF]]
