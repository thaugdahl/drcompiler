// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block{mr=8 nr=16 cache-tile=true mc=256 nc=256 kc=256 l3-size=262144}))' | FileCheck %s

// Versioning a ragged nest is NOT sufficient on its own -- the full tile must
// also be a multiple of the micro-kernel granularity.  The vectorizer's
// `trip % VL != 0` path demands a CONSTANT lower bound, which a `#map(%tileIV)`
// tile bound is not, so a full tile whose extent is not a VL multiple still
// falls back to scalar code.
//
// macroTile clamps to the extent before shrinking (`std::min(nc, N)`), so it
// readily produces such sizes.  Here the raw shrink loop lands on
// (mc, nc, kc) = (100, 110, 60) for this 200x220x240 band under a 256 KiB
// budget -- and 110 % 16 = 14, 60 % 8 = 4.  Stage 1b therefore aligns each tile
// DOWN to its granularity (mc->mr, nc->nr rounded up to a vl multiple, kc->vl)
// before tiling, giving (96, 96, 56).  The shaved-off work is not lost: it
// becomes remainder iterations in the `else` branch rather than scalarizing the
// entire band.

// Separation is over the two spatial tile IVs only (the reduction is left with
// its `to min` bound inside the full tile).
// CHECK:       #[[$SET:.*]] = affine_set<(d0, d1) : (
// CHECK-LABEL: func @gemm
// Aligned tile-space steps: 96 % 8 == 0, 96 % 16 == 0, 56 % 8 == 0.
// CHECK:       affine.for %{{.*}} = 0 to 200 step 96
// CHECK:         affine.for %{{.*}} = 0 to 220 step 96
// CHECK:           affine.for %{{.*}} = 0 to 240 step 56
// CHECK:             affine.if #[[$SET]](%{{.*}}, %{{.*}})
// The full tile register-blocks and vectorizes -- the point of the alignment.
// CHECK:               affine.for %{{.*}} = #{{.*}}(%{{.*}}) to #{{.*}}(%{{.*}}) step 8
// CHECK:                 affine.for %{{.*}} = #{{.*}}(%{{.*}}) to #{{.*}}(%{{.*}}) step 16
// CHECK:                   affine.vector_load {{.*}} vector<8xf64>
// CHECK:             } else {
// CHECK:               affine.for %{{.*}} = #{{.*}}(%{{.*}}) to min #{{.*}}(%{{.*}})
func.func @gemm(%A: memref<200x240xf64>, %B: memref<240x220xf64>, %C: memref<200x220xf64>) {
  affine.for %i = 0 to 200 {
    affine.for %j = 0 to 220 {
      affine.for %k = 0 to 240 {
        %a = affine.load %A[%i, %k] : memref<200x240xf64>
        %b = affine.load %B[%k, %j] : memref<240x220xf64>
        %c = affine.load %C[%i, %j] : memref<200x220xf64>
        %p = arith.mulf %a, %b : f64
        %s = arith.addf %c, %p : f64
        affine.store %s, %C[%i, %j] : memref<200x220xf64>
      }
    }
  }
  return
}
