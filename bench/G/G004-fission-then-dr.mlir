// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission)' | FileCheck %s --check-prefix=FISSION
// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission,data-recomputation{dr-recompute dr-cost-model})' | FileCheck %s --check-prefix=BOTH

// G004: Fission materializes computation into a buffer first.
// Then DR could potentially recognize and eliminate the materialized buffer
// if the chain is cheap enough. Here the chain is expensive (sqrt+div),
// so DR should KEEP the fission buffer.

module {
  func.func @fission_then_dr(%x: memref<?xf64>, %n: index) -> f64 {
    %cst0 = arith.constant 0.0 : f64
    %cst1 = arith.constant 1.0 : f64
    %eps  = arith.constant 0.001 : f64

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

// After fission only: producer loop + buffer created.
// FISSION-LABEL: func.func @fission_then_dr
// FISSION:       memref.alloc
// FISSION:       affine.for
// FISSION:         math.sqrt
// FISSION:         affine.store
// FISSION:       affine.for
// FISSION:         affine.load
// FISSION:       affine.for
// FISSION:         affine.load

// After both: fission buffer survives because the chain is expensive.
// BOTH-LABEL: func.func @fission_then_dr
// BOTH:       memref.alloc
// BOTH:       affine.for
// BOTH:         math.sqrt
// BOTH:         affine.store
// BOTH:       affine.for
// BOTH:         affine.load
