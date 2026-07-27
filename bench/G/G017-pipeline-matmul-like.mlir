// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model})' | FileCheck %s --check-prefix=DR
// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model},memory-fission)' | FileCheck %s --check-prefix=FULL

// G017: Matrix-multiply-like pattern. Initialize a buffer with cheap values,
// compute a matmul-ish accumulation, then reduce.
// Tests the full pipeline on a realistic compute pattern.

module {
  func.func @matmul_like(%n: index) -> f64 {
    %cst0 = arith.constant 0.0 : f64
    %cst1 = arith.constant 1.0 : f64

    // Init buffer: cheap constant store.
    %A = memref.alloc() : memref<64x64xf64>
    %B = memref.alloc() : memref<64x64xf64>
    %C = memref.alloc() : memref<64x64xf64>

    affine.for %i = 0 to 64 {
      affine.for %j = 0 to 64 {
        affine.store %cst1, %A[%i, %j] : memref<64x64xf64>
        affine.store %cst1, %B[%i, %j] : memref<64x64xf64>
        affine.store %cst0, %C[%i, %j] : memref<64x64xf64>
      }
    }

    // Matmul-like: C[i,j] += A[i,k] * B[k,j]
    affine.for %i = 0 to 64 {
      affine.for %j = 0 to 64 {
        affine.for %k = 0 to 64 {
          %a = affine.load %A[%i, %k] : memref<64x64xf64>
          %b = affine.load %B[%k, %j] : memref<64x64xf64>
          %c = affine.load %C[%i, %j] : memref<64x64xf64>
          %prod = arith.mulf %a, %b : f64
          %sum = arith.addf %c, %prod : f64
          affine.store %sum, %C[%i, %j] : memref<64x64xf64>
        }
      }
    }

    // Reduce: sum all C elements.
    %result = affine.for %i = 0 to 64 iter_args(%acc = %cst0) -> (f64) {
      %inner = affine.for %j = 0 to 64 iter_args(%iacc = %acc) -> (f64) {
        %v = affine.load %C[%i, %j] : memref<64x64xf64>
        %s = arith.addf %iacc, %v : f64
        affine.yield %s : f64
      }
      affine.yield %inner : f64
    }

    memref.dealloc %A : memref<64x64xf64>
    memref.dealloc %B : memref<64x64xf64>
    memref.dealloc %C : memref<64x64xf64>
    return %result : f64
  }
}

// With data-recomputation: matmul loads from A and B survive (they are
// loaded inside a loop that also writes C, creating complex provenance).
// DR-LABEL: func.func @matmul_like
// DR:         affine.for
// DR:           affine.for
// DR:             affine.for
// DR:               affine.load
// DR:               arith.mulf

// Full pipeline: same structure, fission has no sibling loops to fission.
// FULL-LABEL: func.func @matmul_like
// FULL:         affine.for
// FULL:           affine.for
// FULL:             affine.for
// FULL:               affine.load
// FULL:               arith.mulf
