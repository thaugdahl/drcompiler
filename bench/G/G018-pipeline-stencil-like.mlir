// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model})' | FileCheck %s --check-prefix=DR
// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission)' | FileCheck %s --check-prefix=FISSION

// G018: 1D stencil with neighbor loads. A[i] = B[i-1] + B[i] + B[i+1].
// B is initialized with IV-dependent values (non-trivial), so loads from B
// cannot be trivially rematerialized.

module {
  func.func @stencil_1d(%input: memref<256xf64>) -> f64 {
    %cst0 = arith.constant 0.0 : f64

    %A = memref.alloc() : memref<256xf64>

    // Stencil: A[i] = input[i-1] + input[i] + input[i+1] for i in [1, 254].
    affine.for %i = 1 to 255 {
      %left   = affine.load %input[%i - 1] : memref<256xf64>
      %center = affine.load %input[%i]     : memref<256xf64>
      %right  = affine.load %input[%i + 1] : memref<256xf64>
      %s1 = arith.addf %left, %center : f64
      %s2 = arith.addf %s1, %right : f64
      affine.store %s2, %A[%i] : memref<256xf64>
    }

    // Reduction over A.
    %result = affine.for %i = 1 to 255 iter_args(%acc = %cst0) -> (f64) {
      %v = affine.load %A[%i] : memref<256xf64>
      %s = arith.addf %acc, %v : f64
      affine.yield %s : f64
    }

    memref.dealloc %A : memref<256xf64>
    return %result : f64
  }
}

// With data-recomputation: stencil loads from input survive (it is a func arg).
// DR-LABEL: func.func @stencil_1d
// DR:         affine.for
// DR:           affine.load
// DR:           arith.addf

// Fission: no sibling loops with shared computation.
// FISSION-LABEL: func.func @stencil_1d
// FISSION:         affine.for
// FISSION:           affine.load
// FISSION:           arith.addf
