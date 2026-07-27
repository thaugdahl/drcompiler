// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model})' | FileCheck %s --check-prefix=DR
// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission)' | FileCheck %s --check-prefix=FISSION

// G006: Same computation chain is targeted by both passes.
// A cheap add chain stored to a buffer (DR can recompute),
// and the same pattern appears in two sibling loops (fission can materialize).
// The buffer version is cheap enough for DR to eliminate.

module {
  func.func @shared_chain(%x: memref<?xf64>, %n: index, %bias: f64) -> f64 {
    %c0 = arith.constant 0 : index
    %cst0 = arith.constant 0.0 : f64
    %cst1 = arith.constant 1.0 : f64

    // Cheap chain stored to buffer -- DR can eliminate.
    %buf = memref.alloc() : memref<1xf64>
    %chain = arith.addf %bias, %cst1 : f64
    memref.store %chain, %buf[%c0] : memref<1xf64>

    // Two sibling loops reading %x with identical cheap computation.
    %sum = affine.for %i = 0 to %n iter_args(%acc = %cst0) -> (f64) {
      %xi = affine.load %x[%i] : memref<?xf64>
      %v = arith.addf %xi, %cst1 : f64
      %out = arith.addf %acc, %v : f64
      affine.yield %out : f64
    }

    %sum2 = affine.for %i = 0 to %n iter_args(%acc = %cst0) -> (f64) {
      %xi = affine.load %x[%i] : memref<?xf64>
      %v = arith.addf %xi, %cst1 : f64
      %out = arith.mulf %acc, %v : f64
      affine.yield %out : f64
    }

    %rv = memref.load %buf[%c0] : memref<1xf64>
    %combined = arith.addf %sum, %sum2 : f64
    %result = arith.addf %combined, %rv : f64
    memref.dealloc %buf : memref<1xf64>
    return %result : f64
  }
}

// DR eliminates the scalar buffer load.
// DR-LABEL: func.func @shared_chain
// DR:         arith.addf
// DR:         affine.for
// DR:         affine.for

// Fission: cheap loop body (add costs 1, not enough for fission threshold).
// Loops remain unchanged, scalar buffer untouched.
// FISSION-LABEL: func.func @shared_chain
// FISSION:       memref.store
// FISSION:       affine.for
// FISSION:         arith.addf
// FISSION:       affine.for
// FISSION:         arith.addf
// FISSION:       memref.load
