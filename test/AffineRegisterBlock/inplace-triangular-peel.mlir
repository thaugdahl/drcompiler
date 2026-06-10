// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block{mr=8 nr=16 vl=8 vectorize=false}))' | FileCheck %s

// In-place triangular reduction with the reduction INNERMOST (PolyBench trmm
// after the alpha-scale is fissioned): B[i][j] += A[k][i] * B[k][j], k=i+1..N.
// The accumulator's memref is read at row k, so the plain alias guard rejects
// jamming the original nest.  The in-place peel splits each mr-strip of i by
// the k range: a sequential CORNER first (k = i2+1 .. ii+8, original
// coordinates -- it reads strip rows and must see pre-MAIN values), then a
// MAIN over k = ii+8..N whose B[k][j] reads are row-disjoint from the strip's
// accumulators (certified, register-blocked: iter_args).  i-trip 44 = 5*8+4:
// epilogue keeps the original nest over [40,44).

// CHECK-LABEL: func.func @trmm_like
// CHECK: affine.for %[[II:.*]] = 0 to 40 step 8 {
// CHECK:   affine.for %[[I2:.*]] = #map{{[0-9]*}}(%[[II]]) to #map{{[0-9]*}}(%[[II]]) {
// CHECK:     affine.for %{{.*}} = 0 to 48 {
// CHECK:       affine.for %{{.*}} = #map{{[0-9]*}}(%[[I2]]) to #map{{[0-9]*}}(%[[II]]) {
// CHECK:   affine.for %{{.*}} = 0 to 48 step 16 {
// CHECK:     affine.for %{{.*}} = #map{{[0-9]*}}(%[[II]]) to 44 iter_args(
// CHECK: affine.for %[[IE:.*]] = 40 to 44 {
// CHECK:   affine.for %{{.*}} = 0 to 48 {
// CHECK:     affine.for %{{.*}} = #map{{[0-9]*}}(%[[IE]]) to 44 {
#map_lb1 = affine_map<(d0) -> (d0 + 1)>
func.func @trmm_like(%A: memref<44x44xf64>, %B: memref<44x48xf64>) {
  affine.for %i = 0 to 44 {
    affine.for %j = 0 to 48 {
      affine.for %k = #map_lb1(%i) to 44 {
        %a = affine.load %A[%k, %i] : memref<44x44xf64>
        %b = affine.load %B[%k, %j] : memref<44x48xf64>
        %m = arith.mulf %a, %b : f64
        %c = affine.load %B[%i, %j] : memref<44x48xf64>
        %s = arith.addf %c, %m : f64
        affine.store %s, %B[%i, %j] : memref<44x48xf64>
      }
    }
  }
  return
}

// LU-shaped in-place factorization must NOT be peeled: the accumulator
// A[i][j] is read at A[i][k] / A[k][j] -- the A[i][k] read varies in k in
// the COLUMN subscript (not the [k][j] row-read shape the peel certifies),
// so the candidate check bails and the alias guard keeps protecting it.

// CHECK-LABEL: func.func @lu_like
// CHECK-NOT: step 8
func.func @lu_like(%A: memref<44x44xf64>) {
  affine.for %i = 0 to 44 {
    affine.for %j = 0 to 44 {
      affine.for %k = #map_lb1(%i) to 44 {
        %a = affine.load %A[%i, %k] : memref<44x44xf64>
        %b = affine.load %A[%k, %j] : memref<44x44xf64>
        %m = arith.mulf %a, %b : f64
        %c = affine.load %A[%i, %j] : memref<44x44xf64>
        %s = arith.subf %c, %m : f64
        affine.store %s, %A[%i, %j] : memref<44x44xf64>
      }
    }
  }
  return
}
