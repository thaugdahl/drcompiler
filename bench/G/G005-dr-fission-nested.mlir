// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model})' | FileCheck %s --check-prefix=DR
// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission)' | FileCheck %s --check-prefix=FISSION
// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model},memory-fission)' | FileCheck %s --check-prefix=BOTH

// G005: Nested pattern where DR handles a cheap inner scalar buffer
// while fission handles shared expensive computation in outer sibling loops.

module {
  func.func @nested(%x: memref<?xf64>, %n: index) -> f64 {
    %c0 = arith.constant 0 : index
    %cst0 = arith.constant 0.0 : f64
    %cst1 = arith.constant 1.0 : f64
    %eps  = arith.constant 0.001 : f64

    // Inner cheap buffer that DR can eliminate.
    %scalar = memref.alloc() : memref<1xf64>
    %sv = arith.addf %cst0, %cst1 : f64
    memref.store %sv, %scalar[%c0] : memref<1xf64>

    // Outer sibling loops with shared expensive computation for fission.
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

    %rv = memref.load %scalar[%c0] : memref<1xf64>
    %combined = arith.addf %sum, %max : f64
    %result = arith.addf %combined, %rv : f64
    memref.dealloc %scalar : memref<1xf64>
    return %result : f64
  }
}

// DR eliminates scalar load, leaves loops.
// DR-LABEL: func.func @nested
// DR:         affine.for
// DR:           math.sqrt
// DR:         affine.for
// DR:           math.sqrt

// Fission creates producer buffer, scalar untouched.
// FISSION-LABEL: func.func @nested
// FISSION:       memref.alloc()
// FISSION:       memref.alloc(
// FISSION:       affine.for
// FISSION:         math.sqrt
// FISSION:         affine.store
// FISSION:       affine.for
// FISSION:         affine.load
// FISSION:       affine.for
// FISSION:         affine.load

// Both: scalar eliminated, fission buffer created.
// BOTH-LABEL: func.func @nested
// BOTH:       memref.alloc
// BOTH:       affine.for
// BOTH:         math.sqrt
// BOTH:         affine.store
// BOTH:       affine.for
// BOTH:         affine.load
