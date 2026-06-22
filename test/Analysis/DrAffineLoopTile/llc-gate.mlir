// Working-set gate (llc-gate): tile a band ONLY when its full footprint exceeds
// the LLC (the kernel is bandwidth-bound).  A 256x256 f32 matmul has a 768 KiB
// footprint (3 * 256^2 * 4 = 786432 B).

// gate=2048 KiB (2 MiB): 768 KiB < 2 MiB -> set fits cache -> SKIP (rationale on
// the band's outer loop; the nest is left untiled).
// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-affine-loop-tile{llc-gate=2048 emit-rationale=true}))' -verify-diagnostics

// gate=2048 KiB: the IR is untouched (no `step` tile loops).
// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-affine-loop-tile{llc-gate=2048}))' | FileCheck %s --check-prefix=SKIP
// SKIP-LABEL: func.func @matmul
// SKIP:         affine.for %{{.*}} = 0 to 256 {
// SKIP-NEXT:      affine.for %{{.*}} = 0 to 256 {
// SKIP-NEXT:        affine.for %{{.*}} = 0 to 256 {
// SKIP-NOT:     step

// gate=256 KiB: 768 KiB > 256 KiB -> spills -> TILE as usual.
// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-affine-loop-tile{llc-gate=256}))' | FileCheck %s --check-prefix=TILE
// TILE-LABEL: func.func @matmul
// TILE:         affine.for %{{.*}} = 0 to 256 step

// gate=0 (default, off): tile regardless of footprint (preserves prior behavior).
// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-affine-loop-tile))' | FileCheck %s --check-prefix=OFF
// OFF-LABEL: func.func @matmul
// OFF:         affine.for %{{.*}} = 0 to 256 step
func.func @matmul(%A: memref<256x256xf32>, %B: memref<256x256xf32>, %C: memref<256x256xf32>) {
  // expected-remark @below {{tile-rationale: SKIP reason=fits-llc footprint=786432 llc-gate=2097152}}
  affine.for %i = 0 to 256 {
    affine.for %k = 0 to 256 {
      affine.for %j = 0 to 256 {
        %a = affine.load %A[%i, %k] : memref<256x256xf32>
        %b = affine.load %B[%k, %j] : memref<256x256xf32>
        %c = affine.load %C[%i, %j] : memref<256x256xf32>
        %m = arith.mulf %a, %b : f32
        %s = arith.addf %c, %m : f32
        affine.store %s, %C[%i, %j] : memref<256x256xf32>
      }
    }
  }
  return
}
