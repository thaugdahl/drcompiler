// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block{mr=8 nr=16 vl=8}))' | FileCheck %s

// Non-mr-divisible triangular bands (correlation shape: trip = N*mr + r,
// r != 0).  The peel must strip-mine only the divisible prefix [lo, stripHi)
// and leave the < mr remainder rows as a scalar epilogue clone of the
// ORIGINAL nest (original coordinates, ragged bound on a real IV, so Stage 3
// provably skips it).

// lb-from-IV orientation (correlation `j = i..M`), i-trip 67 = 8*8 + 3:
// strip covers [0,64) step 8 (DIAG corner + register-blocked HEAD inside),
// epilogue is the original triangular nest over [64,67).

// CHECK-LABEL: func.func @corr_like
// CHECK: affine.for %[[II:.*]] = 0 to 64 step 8 {
// CHECK:   affine.for %{{.*}} = #map{{[0-9]*}}(%[[II]]) to #map{{[0-9]*}}(%[[II]]) {
// CHECK:   affine.for %{{.*}} = #map{{[0-9]*}}(%[[II]]) to 80
// The epilogue keeps the original triangular nest over [64, 67); its
// j-sweep has symbolic-trip bounds (j = i..80), which the affine vl-split
// now vectorizes: a step-8 vector main + scalar tail per row.
// CHECK: affine.for %[[IE:.*]] = 64 to 67 {
// CHECK:   affine.for %{{.*}} = #map{{[0-9]*}}(%[[IE]]) to #map{{[0-9]*}}(%[[IE]]) step 8 {
// CHECK:   affine.for %{{.*}} = #map{{[0-9]*}}(%[[IE]]) to 80 {
// CHECK:     affine.for %{{.*}} = 0 to 128 {
#map_lb = affine_map<(d0) -> (d0)>
func.func @corr_like(%data: memref<128x80xf64>, %corr: memref<80x80xf64>) {
  affine.for %i = 0 to 67 {
    affine.for %j = #map_lb(%i) to 80 {
      affine.for %k = 0 to 128 {
        %a = affine.load %data[%k, %i] : memref<128x80xf64>
        %b = affine.load %data[%k, %j] : memref<128x80xf64>
        %m = arith.mulf %a, %b : f64
        %c = affine.load %corr[%i, %j] : memref<80x80xf64>
        %s = arith.addf %c, %m : f64
        affine.store %s, %corr[%i, %j] : memref<80x80xf64>
      }
    }
  }
  return
}

// Triangular REDUCTION band (trmm shape with a distinct accumulator,
// `k = i+1..N`), i-trip 21 = 2*8 + 5: MAIN/CORNER strip over [0,16),
// epilogue clone of the original i,k,j nest over [16,21).

// CHECK-LABEL: func.func @trmm_like
// CHECK: affine.for %[[II2:.*]] = 0 to 16 step 8 {
// CHECK: affine.for %[[IE2:.*]] = 16 to 21 {
// CHECK-NEXT:   affine.for %{{.*}} = #map{{[0-9]*}}(%[[IE2]]) to 21 {
// CHECK-NEXT:     affine.for %{{.*}} = 0 to 24 {
#map_lb1 = affine_map<(d0) -> (d0 + 1)>
func.func @trmm_like(%A: memref<21x21xf64>, %B: memref<21x24xf64>, %C: memref<21x24xf64>) {
  affine.for %i = 0 to 21 {
    affine.for %k = #map_lb1(%i) to 21 {
      affine.for %j = 0 to 24 {
        %a = affine.load %A[%k, %i] : memref<21x21xf64>
        %b = affine.load %B[%k, %j] : memref<21x24xf64>
        %m = arith.mulf %a, %b : f64
        %c = affine.load %C[%i, %j] : memref<21x24xf64>
        %s = arith.addf %c, %m : f64
        affine.store %s, %C[%i, %j] : memref<21x24xf64>
      }
    }
  }
  return
}
