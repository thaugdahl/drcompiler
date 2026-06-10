// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-affine-loop-tile{cache-size=1 emit-rationale=true}))' -verify-diagnostics

// emit-rationale=true emits a remark on the outermost loop in each band with
// the chosen per-dimension tile sizes and the traffic/footprint breakdown.
// Companion to decision-flip-small-cache.mlir — pins the breakdown so
// cost-model regressions trip lit instead of only runtime.  cache-size=1 KiB
// is clamped to the 4-KiB target floor; the traffic-minimizing search lands
// on 16x16x16 (footprint exactly 4096 B for three f32 tiles).

func.func @matmul(%A: memref<256x256xf32>, %B: memref<256x256xf32>, %C: memref<256x256xf32>) {
  // expected-remark @below {{tile-rationale: TILE sizes=[16, 16, 16] footprint=4096 target=4096 traffic=14417920}}
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
