// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block{cpu-cost-model-file=%S/Inputs/gemm-model.json}))' | FileCheck %s --check-prefix=TILED
// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block))' | FileCheck %s --check-prefix=FLAT

// WP-T3 (TRANSFORMER_KRNL_SPEC): the GEMM configurator drives register-block.
// When a cost-model JSON describes a GEMM model (here arch.fma_units => the
// roofline arm => hasExplicitGemmModel), the cache-tiling decision is made
// per-band by MachineModel::gemmBlocking -- so a deep-K GEMM whose working set
// exceeds the effective L2 auto-cache-tiles (B-panel to L2) WITHOUT the global
// `cache-tile` flag.  The fixture's l2_size=16384 makes this 96 KiB 64x64xf64
// band exceed the effective L2, so it tiles; with NO JSON the GEMM model is
// absent and the band is register-blocked untiled (byte-identical) -- FLAT.

module {
  func.func @gemm(%A: memref<64x64xf64>, %B: memref<64x64xf64>, %C: memref<64x64xf64>) {
    affine.for %i = 0 to 64 {
      affine.for %j = 0 to 64 {
        affine.for %k = 0 to 64 {
          %a = affine.load %A[%i, %k] : memref<64x64xf64>
          %b = affine.load %B[%k, %j] : memref<64x64xf64>
          %c = affine.load %C[%i, %j] : memref<64x64xf64>
          %p = arith.mulf %a, %b : f64
          %s = arith.addf %c, %p : f64
          affine.store %s, %C[%i, %j] : memref<64x64xf64>
        }
      }
    }
    return
  }
}

// Model-driven: the cache-tile loops (stepped by the macro-tile) appear ONLY
// because gemmBlocking flagged the band -- no cache-tile=true on the CLI.
// TILED: affine.for %{{.*}} = 0 to 64 step 16
// TILED:   affine.for %{{.*}} = 0 to 64 step 32
// TILED:     affine.for %{{.*}} = 0 to 64 step 32
// TILED:       affine.for %{{.*}} step 8

// Default machine: no GEMM model => no auto cache-tiling => the outermost loop
// is the register-block i-tile (step 8); the macro-tile step (32) never appears.
// FLAT:     affine.for %{{.*}} = 0 to 64 step 8
// FLAT-NOT: step 32
