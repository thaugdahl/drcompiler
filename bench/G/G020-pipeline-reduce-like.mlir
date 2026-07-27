// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model})' | FileCheck %s --check-prefix=DR
// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission)' | FileCheck %s --check-prefix=FISSION
// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model},memory-fission)' | FileCheck %s --check-prefix=FULL

// G020: Reduction accumulator pattern. Two sibling loops with identical
// expensive computation (sqrt+div) doing different reductions (sum, product).
// Fission should materialize the shared computation.

module {
  func.func @reduce(%x: memref<?xf64>, %n: index) -> f64 {
    %cst0 = arith.constant 0.0 : f64
    %cst1 = arith.constant 1.0 : f64
    %eps  = arith.constant 0.001 : f64

    // Sum reduction with expensive computation.
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

    // Product reduction with same expensive computation.
    %prod = affine.for %i = 0 to %n iter_args(%acc = %cst1) -> (f64) {
      %xi = affine.load %x[%i] : memref<?xf64>
      %sq = arith.mulf %xi, %xi : f64
      %s  = arith.addf %sq, %cst1 : f64
      %r  = math.sqrt %s : f64
      %d  = arith.addf %xi, %eps : f64
      %v  = arith.divf %r, %d : f64
      %out = arith.mulf %acc, %v : f64
      affine.yield %out : f64
    }

    %result = arith.addf %sum, %prod : f64
    return %result : f64
  }
}

// With data-recomputation only: input loads from func arg survive, loops unchanged.
// DR-LABEL: func.func @reduce
// DR:         affine.for
// DR:           math.sqrt
// DR:           arith.divf
// DR:         affine.for
// DR:           math.sqrt
// DR:           arith.divf

// Fission: creates producer loop + buffer, replaces computation in consumers.
// FISSION-LABEL: func.func @reduce
// FISSION:       memref.alloc
// FISSION:       affine.for
// FISSION:         math.sqrt
// FISSION:         arith.divf
// FISSION:         affine.store
// FISSION:       affine.for
// FISSION:         affine.load
// FISSION-NOT:     math.sqrt
// FISSION:         affine.yield
// FISSION:       affine.for
// FISSION:         affine.load
// FISSION-NOT:     math.sqrt
// FISSION:         affine.yield

// Full pipeline: data-recomputation + fission. Fission buffer created.
// FULL-LABEL: func.func @reduce
// FULL:       memref.alloc
// FULL:       affine.for
// FULL:         math.sqrt
// FULL:         affine.store
// FULL:       affine.for
// FULL:         affine.load
// FULL:       affine.for
// FULL:         affine.load
