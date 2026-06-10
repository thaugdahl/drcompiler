// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block{mr=8 nr=16 vl=8 peel-k-tile=64 l3-size=1024}))' | FileCheck %s

// A2: k-chunking of a peeled triangular HEAD.  cov_like is lb-triangular
// with k-STRIDED multiplicands (data[k][i], data[k][j] -- column accesses)
// and a streamed working set (128*64*8 = 64 KB) exceeding the effective LLC
// (l3-size=1024): the HEAD must be wrapped in a kk chunk loop (step = the
// largest divisor of the k-trip <= peel-k-tile = 64; targets < 64 are disabled by the divisor floor), with the inner k
// running [kk, kk+64).  The DIAG stays un-chunked.

// CHECK-LABEL: func.func @cov_like
// CHECK: affine.for %[[II:.*]] = 0 to 64 step 8 {
// CHECK:   affine.for %{{.*}} = #map{{[0-9]*}}(%[[II]]) to #map{{[0-9]*}}(%[[II]]) {
// CHECK:     affine.for %{{.*}} = 0 to 128
// CHECK:   affine.for %[[KK:.*]] = 0 to 128 step 64 {
// CHECK:     affine.for %{{.*}} = #map{{[0-9]*}}(%[[II]]) to 64
// CHECK:       affine.for %{{.*}} = #map{{[0-9]*}}(%[[KK]]) to #map{{[0-9]*}}(%[[KK]])
#map_lb = affine_map<(d0) -> (d0)>
func.func @cov_like(%data: memref<128x64xf64>, %cov: memref<64x64xf64>) {
  affine.for %i = 0 to 64 {
    affine.for %j = #map_lb(%i) to 64 {
      affine.for %k = 0 to 128 {
        %a = affine.load %data[%k, %i] : memref<128x64xf64>
        %b = affine.load %data[%k, %j] : memref<128x64xf64>
        %m = arith.mulf %a, %b : f64
        %c = affine.load %cov[%i, %j] : memref<64x64xf64>
        %s = arith.addf %c, %m : f64
        affine.store %s, %cov[%i, %j] : memref<64x64xf64>
      }
    }
  }
  return
}

// Dot-family triangular band: multiplicands A[i][k] / A[j][k] are STRIDE-1
// in k (row accesses prefetch fine); the gate must NOT chunk k even though
// the footprint exceeds the tiny l3-size.  No step-64 kk loop anywhere.

// CHECK-LABEL: func.func @syrk_like
// CHECK-NOT: step 64
#map_ub = affine_map<(d0) -> (d0 + 1)>
func.func @syrk_like(%A: memref<64x128xf64>, %C: memref<64x64xf64>) {
  affine.for %i = 0 to 64 {
    affine.for %j = 0 to #map_ub(%i) {
      affine.for %k = 0 to 128 {
        %a = affine.load %A[%i, %k] : memref<64x128xf64>
        %b = affine.load %A[%j, %k] : memref<64x128xf64>
        %m = arith.mulf %a, %b : f64
        %c = affine.load %C[%i, %j] : memref<64x64xf64>
        %s = arith.addf %c, %m : f64
        affine.store %s, %C[%i, %j] : memref<64x64xf64>
      }
    }
  }
  return
}
