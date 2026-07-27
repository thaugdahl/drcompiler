// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission{test-diagnostics})' -verify-diagnostics | FileCheck %s
//
// D023: all constants in chain (cost=0) → NOT fission.
// No expensive op (cost >= 15) in the loop body, so no chain is extracted.

module {
  func.func @run(%x: memref<?xf64>, %n: index) -> f64 {
    %cst0 = arith.constant 0.0 : f64
    %cst1 = arith.constant 1.0 : f64
    %cst2 = arith.constant 2.0 : f64
    %cst3 = arith.constant 3.14 : f64

    %sum = affine.for %i = 0 to %n iter_args(%acc = %cst0) -> (f64) {
      %xi = affine.load %x[%i] : memref<?xf64>
      %a  = arith.addf %xi, %cst1 : f64
      %b  = arith.addf %a, %cst2 : f64
      %c  = arith.addf %b, %cst3 : f64
      %out = arith.addf %acc, %c : f64
      affine.yield %out : f64
    }

    %max = affine.for %i = 0 to %n iter_args(%best = %cst0) -> (f64) {
      %xi = affine.load %x[%i] : memref<?xf64>
      %a  = arith.addf %xi, %cst1 : f64
      %b  = arith.addf %a, %cst2 : f64
      %c  = arith.addf %b, %cst3 : f64
      %gt = arith.cmpf ogt, %c, %best : f64
      %out = arith.select %gt, %c, %best : f64
      affine.yield %out : f64
    }

    %result = arith.addf %sum, %max : f64
    return %result : f64
  }
}

// CHECK-LABEL: func.func @run
// No fission — no expensive ops, chain cost < minChainCost.
// CHECK:         affine.for
// CHECK:           arith.addf
// CHECK:         affine.for
// CHECK:           arith.addf
// CHECK-NOT:     memref.alloc
