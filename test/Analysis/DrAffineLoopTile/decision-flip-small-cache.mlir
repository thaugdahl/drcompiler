// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-affine-loop-tile{cache-size=1}))' | FileCheck %s --check-prefix=DRCOMP
// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-loop-tile{cache-size=1}))' | FileCheck %s --check-prefix=UPSTREAM

// Companion to decision-flip-tight-budget.mlir for the tiling pass.  With
// a small cache-size, upstream's nth_root heuristic picks tile sizes of 8
// uniformly across all dims, while the unified-cost-model grid search
// (candidates {2,4,8,16,32,64}) scored by ArchHandler::combineCosts picks
// a different (smaller) outer tile, demonstrating the unified path has
// signal beyond byte-identical reproduction of upstream.

// DRCOMP-LABEL: func.func @matmul
// DRCOMP-DAG: affine.for {{.*}} step 2
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
