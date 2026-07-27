// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model})' | FileCheck %s --check-prefix=DR
// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model},memory-fission)' | FileCheck %s --check-prefix=BOTH

// G003: Data-recomputation runs first, eliminating a cheap scalar load.
// Memory-fission runs second on the remaining loops. The two-pass
// pipeline should produce cleaner output than either pass alone.

module {
  func.func @dr_then_fission(%x: memref<?xf64>, %n: index) -> f64 {
    %c0 = arith.constant 0 : index
    %cst0 = arith.constant 0.0 : f64
    %cst1 = arith.constant 1.0 : f64
    %eps  = arith.constant 0.001 : f64

    // Cheap scalar buffer: DR should eliminate this.
    %scalar = memref.alloc() : memref<1xf64>
    %sval = arith.addf %cst0, %cst1 : f64
    memref.store %sval, %scalar[%c0] : memref<1xf64>

    // Two sibling loops with shared expensive computation from %x.
    // Fission should materialize the shared sqrt+div chain.
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

    %sv = memref.load %scalar[%c0] : memref<1xf64>
    %combined = arith.addf %sum, %max : f64
    %result = arith.addf %combined, %sv : f64
    memref.dealloc %scalar : memref<1xf64>
    return %result : f64
  }
}

// After DR only: scalar load eliminated, loops unchanged.
// DR-LABEL: func.func @dr_then_fission
// DR:         affine.for
// DR:           math.sqrt
// DR:         affine.for
// DR:           math.sqrt

// After both: scalar eliminated, and fission creates producer loop.
// BOTH-LABEL: func.func @dr_then_fission
// BOTH:       memref.alloc
// BOTH:       affine.for
// BOTH:         math.sqrt
// BOTH:         affine.store
// BOTH:       affine.for
// BOTH:         affine.load
// BOTH:       affine.for
// BOTH:         affine.load
