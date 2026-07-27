// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission{test-diagnostics})' -verify-diagnostics | FileCheck %s
//
// D016: scf.for loops instead of affine.for → NOT fission.
// The pass only walks affine.for ops; scf.for is not detected.

module {
  func.func @run(%x: memref<?xf64>, %n: index) -> f64 {
    %cst0 = arith.constant 0.0 : f64
    %cst1 = arith.constant 1.0 : f64
    %c0   = arith.constant 0 : index
    %c1   = arith.constant 1 : index

    %sum = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst0) -> (f64) {
      %xi = memref.load %x[%i] : memref<?xf64>
      %sq = arith.mulf %xi, %xi : f64
      %s  = arith.addf %sq, %cst1 : f64
      %r  = math.sqrt %s : f64
      %out = arith.addf %acc, %r : f64
      scf.yield %out : f64
    }

    %max = scf.for %i = %c0 to %n step %c1 iter_args(%best = %cst0) -> (f64) {
      %xi = memref.load %x[%i] : memref<?xf64>
      %sq = arith.mulf %xi, %xi : f64
      %s  = arith.addf %sq, %cst1 : f64
      %r  = math.sqrt %s : f64
      %gt = arith.cmpf ogt, %r, %best : f64
      %out = arith.select %gt, %r, %best : f64
      scf.yield %out : f64
    }

    %result = arith.addf %sum, %max : f64
    return %result : f64
  }
}

// CHECK-LABEL: func.func @run
// No fission — scf.for not handled.
// CHECK:         scf.for
// CHECK:           math.sqrt
// CHECK:         scf.for
// CHECK:           math.sqrt
// CHECK-NOT:     memref.alloc
