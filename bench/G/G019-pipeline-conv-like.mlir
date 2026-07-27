// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model})' | FileCheck %s --check-prefix=DR
// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission)' | FileCheck %s --check-prefix=FISSION

// G019: 2D convolution pattern. Input is a function arg (no remat possible).
// Kernel is a small 3x3 buffer initialized with constants.
// Output accumulates weighted sums.

module {
  func.func @conv_2d(%input: memref<34x34xf64>, %output: memref<32x32xf64>) {
    %cst0 = arith.constant 0.0 : f64
    %weight = arith.constant 0.111 : f64

    // Init output to zero.
    affine.for %i = 0 to 32 {
      affine.for %j = 0 to 32 {
        affine.store %cst0, %output[%i, %j] : memref<32x32xf64>
      }
    }

    // 2D convolution with 3x3 kernel.
    affine.for %i = 0 to 32 {
      affine.for %j = 0 to 32 {
        affine.for %ki = 0 to 3 {
          affine.for %kj = 0 to 3 {
            %in_val = affine.load %input[%i + %ki, %j + %kj]
                : memref<34x34xf64>
            %weighted = arith.mulf %in_val, %weight : f64
            %cur = affine.load %output[%i, %j] : memref<32x32xf64>
            %new = arith.addf %cur, %weighted : f64
            affine.store %new, %output[%i, %j] : memref<32x32xf64>
          }
        }
      }
    }

    return
  }
}

// With data-recomputation: input loads from func arg survive.
// DR-LABEL: func.func @conv_2d
// DR:         affine.for
// DR:           affine.for
// DR:             affine.for
// DR:               affine.for
// DR:                 affine.load
// DR:                 arith.mulf

// Fission: no sibling loops with matching computation.
// FISSION-LABEL: func.func @conv_2d
// FISSION:       affine.for
// FISSION:         affine.for
// FISSION:           affine.for
// FISSION:             affine.for
// FISSION:               affine.load
