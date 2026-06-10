// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block{mr=8 nr=16 vl=8}))' | FileCheck %s

// lb-from-IV triangular band (covariance/correlation shape, `j = i..M`):
// peel must strip-mine i by mr, emit the ragged DIAG corner in ORIGINAL
// coordinates (for i2 = ii..ii+mr { for j = i2..ii+mr }) so Stage 3 cannot
// unroll-jam it into an empty range, and register-block the rectangular
// HEAD (j = ii+mr..M).

// CHECK-LABEL: func.func @cov_like
// CHECK: affine.for %[[II:.*]] = 0 to 64 step 8 {
// CHECK:   affine.for %[[I2:.*]] = #map(%[[II]]) to #map1(%[[II]]) {
// CHECK:   affine.for %{{.*}} = #map1(%[[II]]) to 64
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

// Imperfect spatial body: the j-loop carries scalar statements (the mean
// pattern `m[j]=0; for i acc; m[j]/=n`).  The broadcast vectorizer must NOT
// re-step j — that executes the statements once per VL lanes (measured on
// covariance: only every 16th mean initialized/divided).  Everything stays
// scalar (no vector ops, no step on j).

// CHECK-LABEL: func.func @mean_like
// CHECK-NOT: vector
// CHECK: affine.for %{{.*}} = 0 to 64 {
// CHECK-NOT: step
func.func @mean_like(%data: memref<128x64xf64>, %mean: memref<64xf64>, %n: f64) {
  %cst = arith.constant 0.0 : f64
  affine.for %j = 0 to 64 {
    affine.store %cst, %mean[%j] : memref<64xf64>
    affine.for %i = 0 to 128 {
      %d = affine.load %data[%i, %j] : memref<128x64xf64>
      %m = affine.load %mean[%j] : memref<64xf64>
      %s = arith.addf %m, %d : f64
      affine.store %s, %mean[%j] : memref<64xf64>
    }
    %m2 = affine.load %mean[%j] : memref<64xf64>
    %dv = arith.divf %m2, %n : f64
    affine.store %dv, %mean[%j] : memref<64xf64>
  }
  return
}
