// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model})' | FileCheck %s --check-prefix=DR
// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission{test-diagnostics})' | FileCheck %s --check-prefix=FISSION

// G002: Data-recomputation says KEEP (expensive sqrt+div chain with 2 consumers),
// but memory-fission says FISSION (materializes the shared expensive computation
// into a buffer to avoid redundant recomputation across sibling loops).

module {
  func.func @disagree(%x: memref<?xf64>, %n: index) -> f64 {
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

// Data-recomputation: no buffers to analyze (these are memref<?xf64> args),
// pass is essentially a no-op for this pattern. Loops remain unchanged.
// DR-LABEL: func.func @disagree
// DR:         affine.for
// DR:           math.sqrt
// DR:         affine.for
// DR:           math.sqrt

// Fission: creates producer loop + buffer, replaces computation in consumers.
// FISSION-LABEL: func.func @disagree
// FISSION:       memref.alloc
// FISSION:       affine.for
// FISSION:         math.sqrt
// FISSION:         affine.store
// FISSION:       affine.for
// FISSION:         affine.load
// FISSION-NOT:     math.sqrt
// FISSION:         affine.yield
// FISSION:       affine.for
// FISSION:         affine.load
// FISSION-NOT:     math.sqrt
// FISSION:         affine.yield
