// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-affine-stencil-time-tile{force-tile=true tile-t=4 tile-s=8}))' | FileCheck %s
// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-affine-stencil-time-tile))' | FileCheck %s --check-prefix=GATE

// WP3.3 (COSTMODEL_V4_SPEC §4.3): fdtd-2d is a time loop with FOUR phases over
// three 2-D arrays (ey, ex, hz) + a 1-D source (fict): a 1-D border phase
// (ey[0][j] = fict[t]) then three 2-D stencils, all halo <= 1.  Inter-phase
// dependences have virtual-time distance >= 1, so the tau-ONLY skew (i'=i+tau,
// j'=j+tau, tau = 4t + phase, c=0) makes every distance non-negative.  Tiled
// into tt / ii / jj (jj independent of ii since c=0) / t' / per-phase (i,j)
// windows.  Validated bit-identical at SMALL; XL 2.51x (beats Polly 1.35x).

// CHECK-LABEL: func.func @fdtd
// CHECK: affine.for %[[TT:.*]] = 0 to 40 step 4 {
// CHECK:   affine.for %[[II:.*]] = #map{{[0-9]*}}(%[[TT]]) to #map{{[0-9]*}}(%[[TT]]) step 8 {
// CHECK:     affine.for %[[JJ:.*]] = #map{{[0-9]*}}(%[[TT]]) to #map{{[0-9]*}}(%[[TT]]) step 8 {
// CHECK:       affine.for %[[T:.*]] = #map{{[0-9]*}}(%[[TT]]) to min #map{{[0-9]*}}(%[[TT]]) {
// Phase 0 (1-D border, modelled as a degenerate 2-D phase): writes row 0 of ey.
// CHECK:         affine.for %{{.*}} = max #map{{[0-9]*}}(%[[II]], %[[T]]) to min #map{{[0-9]*}}(%[[II]], %[[T]]) {
// CHECK:           affine.for %{{.*}} = max #map{{[0-9]*}}(%[[JJ]], %[[T]]) to min #map{{[0-9]*}}(%[[JJ]], %[[T]]) {
// CHECK:             affine.store %{{.*}}, %{{.*}}[0, %{{.*}}]
// Three more 2-D phase pairs follow (one per remaining phase).
// CHECK-COUNT-3: affine.for %{{.*}} = max #map{{[0-9]*}}(%[[II]], %[[T]]) to min #map{{[0-9]*}}(%[[II]], %[[T]]) {

// At 64x64 the gate refuses (3 arrays x 64^2 x 8 B is cache-resident).
// GATE-LABEL: func.func @fdtd
// GATE: affine.for %{{.*}} = 0 to 40 {
// GATE-NOT: step 4

func.func @fdtd(%ex: memref<64x64xf64>, %ey: memref<64x64xf64>, %hz: memref<64x64xf64>, %fict: memref<64xf64>) {
  %c5 = arith.constant 0.5 : f64
  %c7 = arith.constant 0.7 : f64
  affine.for %t = 0 to 40 {
    affine.for %j = 0 to 64 {
      %f = affine.load %fict[%t] : memref<64xf64>
      affine.store %f, %ey[0, %j] : memref<64x64xf64>
    }
    affine.for %i = 1 to 64 {
      affine.for %j = 0 to 64 {
        %a = affine.load %ey[%i, %j] : memref<64x64xf64>
        %b = affine.load %hz[%i, %j] : memref<64x64xf64>
        %c = affine.load %hz[%i - 1, %j] : memref<64x64xf64>
        %d = arith.subf %b, %c : f64
        %e = arith.mulf %d, %c5 : f64
        %g = arith.subf %a, %e : f64
        affine.store %g, %ey[%i, %j] : memref<64x64xf64>
      }
    }
    affine.for %i = 0 to 64 {
      affine.for %j = 1 to 64 {
        %a = affine.load %ex[%i, %j] : memref<64x64xf64>
        %b = affine.load %hz[%i, %j] : memref<64x64xf64>
        %c = affine.load %hz[%i, %j - 1] : memref<64x64xf64>
        %d = arith.subf %b, %c : f64
        %e = arith.mulf %d, %c5 : f64
        %g = arith.subf %a, %e : f64
        affine.store %g, %ex[%i, %j] : memref<64x64xf64>
      }
    }
    affine.for %i = 0 to 63 {
      affine.for %j = 0 to 63 {
        %a = affine.load %hz[%i, %j] : memref<64x64xf64>
        %b = affine.load %ex[%i, %j + 1] : memref<64x64xf64>
        %c = affine.load %ex[%i, %j] : memref<64x64xf64>
        %d = arith.subf %b, %c : f64
        %e = affine.load %ey[%i + 1, %j] : memref<64x64xf64>
        %f2 = affine.load %ey[%i, %j] : memref<64x64xf64>
        %g = arith.addf %d, %e : f64
        %h = arith.subf %g, %f2 : f64
        %k = arith.mulf %h, %c7 : f64
        %l = arith.subf %a, %k : f64
        affine.store %l, %hz[%i, %j] : memref<64x64xf64>
      }
    }
  }
  return
}
