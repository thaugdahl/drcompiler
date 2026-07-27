// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission{test-diagnostics})' -verify-diagnostics | FileCheck %s
//
// D004: add-only chain (cost<15) in 2 loops → should NOT fission.
// Chain: mulf(3) + addf(1) + addf(1) + addf(1) + mulf(3) = 9.
// 9 < minChainCost=15, so no chain is extracted, no candidate formed.

module {
  func.func @run(%x: memref<?xf64>, %n: index) -> f64 {
    %cst0 = arith.constant 0.0 : f64
    %cst1 = arith.constant 1.0 : f64
    %cst2 = arith.constant 2.0 : f64

    %sum = affine.for %i = 0 to %n iter_args(%acc = %cst0) -> (f64) {
      %xi = affine.load %x[%i] : memref<?xf64>
      %a  = arith.mulf %xi, %xi : f64
      %b  = arith.addf %a, %cst1 : f64
      %c  = arith.addf %b, %cst2 : f64
      %d  = arith.addf %c, %xi : f64
      %e  = arith.mulf %d, %cst2 : f64
      %out = arith.addf %acc, %e : f64
      affine.yield %out : f64
    }

    %max = affine.for %i = 0 to %n iter_args(%best = %cst0) -> (f64) {
      %xi = affine.load %x[%i] : memref<?xf64>
      %a  = arith.mulf %xi, %xi : f64
      %b  = arith.addf %a, %cst1 : f64
      %c  = arith.addf %b, %cst2 : f64
      %d  = arith.addf %c, %xi : f64
      %e  = arith.mulf %d, %cst2 : f64
      %gt = arith.cmpf ogt, %e, %best : f64
      %out = arith.select %gt, %e, %best : f64
      affine.yield %out : f64
    }

    %result = arith.addf %sum, %max : f64
    return %result : f64
  }
}

// CHECK-LABEL: func.func @run
// No fission — cheap chain, computation stays inline.
// CHECK:         affine.for
// CHECK:           arith.mulf
// CHECK:           arith.addf
// CHECK:         affine.for
// CHECK:           arith.mulf
// CHECK:           arith.addf
// CHECK-NOT:     memref.alloc
