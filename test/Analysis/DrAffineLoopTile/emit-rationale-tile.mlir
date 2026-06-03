// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-affine-loop-tile{cache-size=1 emit-rationale=true}))' -verify-diagnostics

// emit-rationale=true emits a remark on the outermost loop in each band with
// the chosen tile size and the tile-sensitive cost breakdown (memory-reuse
// traffic vs register-block spill).  Companion to decision-flip-small-cache.mlir
// — pins the breakdown so cost-model regressions trip lit instead of only
// runtime.  With the generic register budget (regCap=32) the reuse/register
// tension lands the uniform tile at size=4 here.

func.func @matmul(%A: memref<256x256xf32>, %B: memref<256x256xf32>, %C: memref<256x256xf32>) {
  // expected-remark @below {{tile-rationale: TILE size=4 tile_cost=16777216 untiled_cost=67108864}}
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
