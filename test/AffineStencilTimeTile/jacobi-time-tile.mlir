// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-affine-stencil-time-tile{force-tile=true tile-t=8 tile-s=32}))' | FileCheck %s
// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-affine-stencil-time-tile))' | FileCheck %s --check-prefix=GATE

// B2: skewed time-tiling of the ping-pong stencil shape
//   for t { B <- f(A); A <- g(B) }.
// Tiled form: tt (step tile-t), skewed space tiles ii (step tile-s, range
// extended by 2*T), inner t clipped to the tile, then the two phase nests
// with windowed bounds lb = max(lo, ii - 2t - phase),
// ub = min(hi, ii + Ts - 2t - phase).  Bounds hang directly off the real
// IVs (maps, no affine.apply between loops).

// CHECK-LABEL: func.func @jacobi1d_like
// CHECK: affine.for %[[TT:.*]] = 0 to 50 step 8 {
// CHECK:   affine.for %[[II:.*]] = #map{{[0-9]*}}(%[[TT]]) to #map{{[0-9]*}}(%[[TT]]) step 32 {
// CHECK:     affine.for %[[T:.*]] = #map{{[0-9]*}}(%[[TT]]) to min #map{{[0-9]*}}(%[[TT]]) {
// CHECK:       affine.for %{{.*}} = max #map{{[0-9]*}}(%[[II]], %[[T]]) to min #map{{[0-9]*}}(%[[II]], %[[T]]) {
// CHECK:         affine.load %arg0
// CHECK:         affine.store %{{.*}}, %arg1
// CHECK:       affine.for %{{.*}} = max #map{{[0-9]*}}(%[[II]], %[[T]]) to min #map{{[0-9]*}}(%[[II]], %[[T]]) {
// CHECK:         affine.load %arg1
// CHECK:         affine.store %{{.*}}, %arg0

// The footprint gate must refuse the same kernel un-forced: 2 arrays x 100
// doubles is laughably cache-resident (this is Polly's jacobi-1d 0.34x
// mistake).  Original loop structure intact.
// GATE-LABEL: func.func @jacobi1d_like
// GATE: affine.for %{{.*}} = 0 to 50 {
// GATE-NOT: step 8
func.func @jacobi1d_like(%A: memref<100xf64>, %B: memref<100xf64>) {
  %c = arith.constant 0.33333 : f64
  affine.for %t = 0 to 50 {
    affine.for %i = 1 to 99 {
      %0 = affine.load %A[%i - 1] : memref<100xf64>
      %1 = affine.load %A[%i] : memref<100xf64>
      %2 = arith.addf %0, %1 : f64
      %3 = affine.load %A[%i + 1] : memref<100xf64>
      %4 = arith.addf %2, %3 : f64
      %5 = arith.mulf %4, %c : f64
      affine.store %5, %B[%i] : memref<100xf64>
    }
    affine.for %i = 1 to 99 {
      %0 = affine.load %B[%i - 1] : memref<100xf64>
      %1 = affine.load %B[%i] : memref<100xf64>
      %2 = arith.addf %0, %1 : f64
      %3 = affine.load %B[%i + 1] : memref<100xf64>
      %4 = arith.addf %2, %3 : f64
      %5 = arith.mulf %4, %c : f64
      affine.store %5, %A[%i] : memref<100xf64>
    }
  }
  return
}

// In-place Gauss-Seidel (seidel-2d): a SINGLE nest updating its own array.
// Not the ping-pong shape; B2's tiling is illegal here -- must not fire
// even when forced.
// CHECK-LABEL: func.func @seidel_like
// CHECK-NOT: max #map
func.func @seidel_like(%A: memref<100x100xf64>) {
  %c = arith.constant 0.111 : f64
  affine.for %t = 0 to 50 {
    affine.for %i = 1 to 99 {
      affine.for %j = 1 to 99 {
        %0 = affine.load %A[%i - 1, %j] : memref<100x100xf64>
        %1 = affine.load %A[%i, %j - 1] : memref<100x100xf64>
        %2 = arith.addf %0, %1 : f64
        %3 = affine.load %A[%i, %j + 1] : memref<100x100xf64>
        %4 = arith.addf %2, %3 : f64
        %5 = arith.mulf %4, %c : f64
        affine.store %5, %A[%i, %j] : memref<100x100xf64>
      }
    }
  }
  return
}
