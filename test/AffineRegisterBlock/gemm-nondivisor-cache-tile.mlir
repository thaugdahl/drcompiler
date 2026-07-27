// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block{mr=8 nr=16 cache-tile=true mc=64 nc=64 kc=64 l3-size=65536}))' | FileCheck %s

// The cost model sizes a macro-tile for CAPACITY, so it has no reason to land on
// a divisor of the extent -- PolyBench's LARGE sizes are deliberately non-round
// (gemm 1000x1100x1200 against a 128x256x256 tile divides in no dimension).
//
// A non-divisor tile makes tilePerfectlyNested emit intra-tile loops bounded by
// `to min`, whose bound map has two results.  getConstantTripCount then returns
// nullopt, loopUnrollJamByFactor rejects the band, and Stage 3 drops it -- the
// whole GEMM falls back to scalar.  Stage 1b therefore versions a ragged nest
// into a full-tile clone (single-result bound maps, constant trip) plus the
// original ragged nest, via separateFullTiles.
//
// Versioning is over the SPATIAL loops (i, j) only -- the reduction (k) is left
// with its `to min` bound INSIDE the full tile, because k is never unroll-jammed
// so a min bound does not block register blocking; separating it too would push
// the k-remainder panel into the scalar else branch for nothing.  So the
// separation condition is a 2-D affine_set over (i_tile, j_tile).
//
// Both functions below tile to 32x64x64 under this 64 KiB budget (the shrink
// loop halves mc once: (32*64 + 64*64 + 32*64)*8 == 65536).

// -----------------------------------------------------------------------------
// RAGGED: 100 % 32 = 4, 100 % 64 = 36 -> versioned.
// -----------------------------------------------------------------------------
// The separation guard is a 2-D set over the spatial tile IVs only (i, j) --
// the reduction is not versioned.  (Printed at top level, before the func.)
// CHECK:       #[[$SET:.*]] = affine_set<(d0, d1) : (
// CHECK-LABEL: func @gemm_ragged
func.func @gemm_ragged(%A: memref<100x100xf64>, %B: memref<100x100xf64>, %C: memref<100x100xf64>) {
  // The tile-space loops survive, stepped by the (aligned) tile size.
  // CHECK:       affine.for %{{.*}} = 0 to 100 step 32
  // CHECK:         affine.for %{{.*}} = 0 to 100 step 64
  // CHECK:           affine.for %{{.*}} = 0 to 100 step 64
  //
  // The two-version guard over (i_tile, j_tile), and the FULL-TILE branch:
  // spatial bounds are single-result (no `to min`), so the register-block
  // micro-kernel applies -- mr=8 stepping and real vector ops.
  // CHECK:             affine.if #[[$SET]](%{{.*}}, %{{.*}})
  // CHECK:               affine.for %{{.*}} = #{{.*}}(%{{.*}}) to #{{.*}}(%{{.*}}) step 8
  // CHECK:                 affine.for %{{.*}} = #{{.*}}(%{{.*}}) to #{{.*}}(%{{.*}}) step 16
  // CHECK:                   affine.vector_load {{.*}} vector<8xf64>
  //
  // The REMAINDER branch keeps the ragged `to min` form and stays scalar; it is
  // self-filtering (loopUnrollJamByFactor fails on a two-result bound map).
  // CHECK:             } else {
  // CHECK:               affine.for %{{.*}} = #{{.*}}(%{{.*}}) to min #{{.*}}(%{{.*}})
  affine.for %i = 0 to 100 {
    affine.for %j = 0 to 100 {
      affine.for %k = 0 to 100 {
        %a = affine.load %A[%i, %k] : memref<100x100xf64>
        %b = affine.load %B[%k, %j] : memref<100x100xf64>
        %c = affine.load %C[%i, %j] : memref<100x100xf64>
        %p = arith.mulf %a, %b : f64
        %s = arith.addf %c, %p : f64
        affine.store %s, %C[%i, %j] : memref<100x100xf64>
      }
    }
  }
  return
}

// -----------------------------------------------------------------------------
// DIVISOR: 128 % 32 == 0 and 128 % 64 == 0 -> must NOT be versioned.
//
// separateFullTiles does not decline on a divisor-sized tile: it emits a
// tautological `affine.if () : (0 == 0)` and duplicates the body into both
// regions.  Stage 1b runs no canonicalizer before Stage 2, so that doubled IR
// would reach the register-blocker on cases that already work -- hence the
// explicit ragged gate.
// -----------------------------------------------------------------------------
// CHECK-LABEL: func @gemm_divisor
// CHECK-NOT:     affine.if
// CHECK-NOT:     to min
// CHECK:         affine.vector_load {{.*}} vector<8xf64>
func.func @gemm_divisor(%A: memref<128x128xf64>, %B: memref<128x128xf64>, %C: memref<128x128xf64>) {
  affine.for %i = 0 to 128 {
    affine.for %j = 0 to 128 {
      affine.for %k = 0 to 128 {
        %a = affine.load %A[%i, %k] : memref<128x128xf64>
        %b = affine.load %B[%k, %j] : memref<128x128xf64>
        %c = affine.load %C[%i, %j] : memref<128x128xf64>
        %p = arith.mulf %a, %b : f64
        %s = arith.addf %c, %p : f64
        affine.store %s, %C[%i, %j] : memref<128x128xf64>
      }
    }
  }
  return
}
