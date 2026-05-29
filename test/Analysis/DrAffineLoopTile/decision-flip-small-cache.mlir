// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-affine-loop-tile{cache-size=1}))' | FileCheck %s --check-prefix=DRCOMP
// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-loop-tile{cache-size=1}))' | FileCheck %s --check-prefix=UPSTREAM

// Companion to decision-flip-tight-budget.mlir for the tiling pass.  With
// a tiny 1-KiB cache, upstream's nth_root heuristic picks uniform 8 across
// all 3 dims regardless.  drcomp's per-problem cost model sees that
// num_tiles * per_tile_mem exceeds the untiled baseline at every outer
// candidate and falls back to tile=1 for the outer dims, only emitting
// the balance step on the innermost dim.  The divergence demonstrates
// the unified path makes a meaningfully different decision than
// upstream's placeholder.

// DRCOMP-LABEL: func.func @matmul
// DRCOMP: affine.for {{.*}} step 128
// DRCOMP: return

// UPSTREAM-LABEL: func.func @matmul
// UPSTREAM-DAG: affine.for {{.*}} step 8
// UPSTREAM: return

func.func @matmul(%A: memref<256x256xf32>, %B: memref<256x256xf32>, %C: memref<256x256xf32>) {
  affine.for %i = 0 to 256 {
    affine.for %j = 0 to 256 {
      affine.for %k = 0 to 256 {
        %a = affine.load %A[%i, %k] : memref<256x256xf32>
        %b = affine.load %B[%k, %j] : memref<256x256xf32>
        %c = affine.load %C[%i, %j] : memref<256x256xf32>
        %mul = arith.mulf %a, %b : f32
        %add = arith.addf %c, %mul : f32
        affine.store %add, %C[%i, %j] : memref<256x256xf32>
      }
    }
  }
  return
}
