// Under a GEMM cost-model, affine-register-block both vectorizes the GEMM
// (canonicalizeAllocaGemm + Stage 2) AND cache-tiles it.  The cache-tile loops
// (K-tile reduction + packed/C-tile scratch) introduce carried dependences that
// make par-spmd-perband bail the whole GEMM to par.critical (serial) -- which
// capped the openai-gpt vec+par compose.  `no-cache-tile` keeps the vectorization
// but skips the tiling, leaving the simple stepped register-blocked band that
// SHARDS into a par.forall.  See PARALLEL_SPMD_SPEC.md §11.18.
//
// The fixture sets llc_sharers=32 so the 3 MiB (512^3 f32) working set exceeds
// the leniency-derated per-worker LLC share (128 MiB / 32 * 0.5 = 2 MiB) and
// therefore tiles -- which is the realistic SPMD picture: the workers that would
// shard this band all contend for the same LLC.  As sole occupant this working
// set fits the LLC and gemmBlocking would (correctly) not tile it at all.

// Default (model, cache-tiled): the GEMM gets a cache-tile loop and does NOT shard.
// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block{mr=8 nr=16 cpu-cost-model-file=%S/Inputs/zen4-gemm.json}),dr-par-bubbles{par-spmd-perband})' 2>&1 | FileCheck %s --check-prefix=TILED

// no-cache-tile: vectorized AND sharded (par.forall, no par.critical).
// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block{mr=8 nr=16 cpu-cost-model-file=%S/Inputs/zen4-gemm.json no-cache-tile}),dr-par-bubbles{par-spmd-perband})' 2>&1 | FileCheck %s --check-prefix=SHARD

// TILED: par-spmd-perband: materialized foralls=0 critical=1
// SHARD: par-spmd-perband: materialized foralls=1 critical=0
func.func @gemm(%A: memref<512x512xf32>, %B: memref<512x512xf32>, %C: memref<512x512xf32>) {
  affine.for %i = 0 to 512 {
    affine.for %j = 0 to 512 {
      affine.for %k = 0 to 512 {
        %a = affine.load %A[%i, %k] : memref<512x512xf32>
        %b = affine.load %B[%k, %j] : memref<512x512xf32>
        %c = affine.load %C[%i, %j] : memref<512x512xf32>
        %p = arith.mulf %a, %b : f32
        %s = arith.addf %c, %p : f32
        affine.store %s, %C[%i, %j] : memref<512x512xf32>
      }
    }
  }
  return
}
