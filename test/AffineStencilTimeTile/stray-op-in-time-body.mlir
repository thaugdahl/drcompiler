// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-affine-stencil-time-tile{force-tile=true tile-t=8 tile-s=32}))' | FileCheck %s

// The classifier admits a time loop whose body holds exactly TWO affine.for
// children *plus any number of memory-effect-free ops* -- here an `arith.constant`
// sitting BETWEEN the two phases.  timeTile must re-filter the body for
// AffineForOp instead of casting every op, or the phase roots come out as
// {phase1, constant} and the ping-pong shape is missed (an assert-enabled build
// aborts outright in the cast).
//
// This is the jacobi-1d ping-pong of jacobi-time-tile.mlir, so with the roots
// picked correctly it must tile exactly the same way.

// CHECK-LABEL: func.func @jacobi1d_stray_op
// CHECK: affine.for %[[TT:.*]] = 0 to 50 step 8 {
// CHECK:   affine.for %[[II:.*]] = #map{{[0-9]*}}(%[[TT]]) to #map{{[0-9]*}}(%[[TT]]) step 32 {
// CHECK:     affine.for %[[T:.*]] = #map{{[0-9]*}}(%[[TT]]) to min #map{{[0-9]*}}(%[[TT]]) {
// CHECK:       affine.for %{{.*}} = max #map{{[0-9]*}}(%[[II]], %[[T]]) to min #map{{[0-9]*}}(%[[II]], %[[T]]) {
// CHECK:         affine.load %arg0
// CHECK:         affine.store %{{.*}}, %arg1
// CHECK:       affine.for %{{.*}} = max #map{{[0-9]*}}(%[[II]], %[[T]]) to min #map{{[0-9]*}}(%[[II]], %[[T]]) {
// CHECK:         affine.load %arg1
// CHECK:         affine.store %{{.*}}, %arg0
func.func @jacobi1d_stray_op(%A: memref<100xf64>, %B: memref<100xf64>) {
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
    %stray = arith.constant 0.5 : f64
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
