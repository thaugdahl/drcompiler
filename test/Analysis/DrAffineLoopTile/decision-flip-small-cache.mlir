// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-affine-loop-tile{cache-size=1}))' | FileCheck %s --check-prefix=DRCOMP
// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-loop-tile{cache-size=1}))' | FileCheck %s --check-prefix=UPSTREAM

// Companion to decision-flip-tight-budget.mlir for the tiling pass.  With a
// tiny 1-KiB cache, upstream's nth_root heuristic picks uniform 8 across all 3
// dims.  drcomp's unified cost model ranks tile sizes by the tile-sensitive
// cost (memory-reuse traffic traded against the register-block spill that
// grows with the c^2 accumulator block) and emits its choice uniformly.  At
// the generic register budget that tension lands the tile at 4, not 8 — a
// meaningfully different decision from upstream's placeholder.

// DRCOMP-LABEL: func.func @matmul
// DRCOMP: affine.for {{.*}} step 4
// DRCOMP: affine.for {{.*}} step 4
// DRCOMP: affine.for {{.*}} step 4
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
