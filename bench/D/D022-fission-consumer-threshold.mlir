// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission{test-diagnostics})' -verify-diagnostics | FileCheck %s
//
// D022: chain at exactly minChainCost=15 (single divf).
// Chain: divf(15) + constant(0) = 15.  Exactly at threshold.
// Tip = divf (user addf cost 1 < minConsumerCost=10).
// recompute = 2*15 = 30, keep = 15 + 1 + 2*4 = 24 → FISSION.

module {
  func.func @run(%x: memref<?xf64>, %n: index) -> f64 {
    %cst0 = arith.constant 0.0 : f64
    %cst1 = arith.constant 1.0 : f64

    // expected-remark @below {{memory-fission: FISSION (compute=15, consumers=2)}}
    // expected-remark @below {{materialized buffer for 2 consumers}}
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
