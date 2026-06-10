// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-test-reuse-analysis{cache-bytes=32768}))' -verify-diagnostics

// B1: stencil-neighbour reference GROUPS.  Per-reference analysis sees
// A[i][j-1] / A[i][j+1] as unrelated streams (no temporal reuse anywhere);
// the group view records that they share one coefficient matrix and differ
// only in constant offsets: span = halo width per dim, carries[l] = loop l
// indexes a dim whose offsets differ (one l-iteration re-touches a
// neighbour's element).

// jacobi-2d 5-point space nest: one group of 5 loads on A, span [2,2],
// both space loops carry group reuse.  B's store is a singleton (groups
// with < 2 members are not recorded).
func.func @jacobi2d_step(%A: memref<1300x1300xf64>, %B: memref<1300x1300xf64>) {
  %c = arith.constant 2.000000e-01 : f64
  // expected-remark @below {{reuse-analysis: band depth=2 trips=[1298, 1298] footprint=80870592 evictedReuse=[0,0] anyTemporal=0}}
  affine.for %i = 1 to 1299 {
    affine.for %j = 1 to 1299 {
      // expected-remark @below {{reuse-analysis: group members=5 span=[2, 2] carries=[1,1]}}
      // expected-remark @below {{reuse-analysis: ref load x1 kinds=[stream,spat] iterFP=[10384,8]}}
      %0 = affine.load %A[%i, %j] : memref<1300x1300xf64>
      // expected-remark @below {{reuse-analysis: ref load x1 kinds=[stream,spat] iterFP=[10384,8]}}
      %1 = affine.load %A[%i, %j - 1] : memref<1300x1300xf64>
      // expected-remark @below {{reuse-analysis: ref load x1 kinds=[stream,spat] iterFP=[10384,8]}}
      %2 = affine.load %A[%i, %j + 1] : memref<1300x1300xf64>
      // expected-remark @below {{reuse-analysis: ref load x1 kinds=[stream,spat] iterFP=[10384,8]}}
      %3 = affine.load %A[%i + 1, %j] : memref<1300x1300xf64>
      // expected-remark @below {{reuse-analysis: ref load x1 kinds=[stream,spat] iterFP=[10384,8]}}
      %4 = affine.load %A[%i - 1, %j] : memref<1300x1300xf64>
      %s1 = arith.addf %0, %1 : f64
      %s2 = arith.addf %s1, %2 : f64
      %s3 = arith.addf %s2, %3 : f64
      %s4 = arith.addf %s3, %4 : f64
      %r = arith.mulf %s4, %c : f64
      // expected-remark @below {{reuse-analysis: ref store x1 kinds=[stream,spat] iterFP=[10384,8]}}
      affine.store %r, %B[%i, %j] : memref<1300x1300xf64>
    }
  }
  return
}

// heat-3d 7-point: one group of 7 on A, span [2,2,2], all three space loops
// carry.  (The two i±1 / j±1 / k±1 pairs all share the centre point's
// coefficient matrix.)
func.func @heat3d_step(%A: memref<120x120x120xf64>, %B: memref<120x120x120xf64>) {
  %c = arith.constant 1.250000e-01 : f64
  // expected-remark @below {{reuse-analysis: band depth=3 trips=[118, 118, 118] footprint=105154048 evictedReuse=[0,0,0] anyTemporal=0}}
  affine.for %i = 1 to 119 {
    affine.for %j = 1 to 119 {
      affine.for %k = 1 to 119 {
        // expected-remark @below {{reuse-analysis: group members=7 span=[2, 2, 2] carries=[1,1,1]}}
        // expected-remark @below {{reuse-analysis: ref load x1 kinds=[stream,stream,spat] iterFP=[111392,944,8]}}
        %0 = affine.load %A[%i, %j, %k] : memref<120x120x120xf64>
        // expected-remark @below {{reuse-analysis: ref load x1 kinds=[stream,stream,spat] iterFP=[111392,944,8]}}
        %1 = affine.load %A[%i - 1, %j, %k] : memref<120x120x120xf64>
        // expected-remark @below {{reuse-analysis: ref load x1 kinds=[stream,stream,spat] iterFP=[111392,944,8]}}
        %2 = affine.load %A[%i + 1, %j, %k] : memref<120x120x120xf64>
        // expected-remark @below {{reuse-analysis: ref load x1 kinds=[stream,stream,spat] iterFP=[111392,944,8]}}
        %3 = affine.load %A[%i, %j - 1, %k] : memref<120x120x120xf64>
        // expected-remark @below {{reuse-analysis: ref load x1 kinds=[stream,stream,spat] iterFP=[111392,944,8]}}
        %4 = affine.load %A[%i, %j + 1, %k] : memref<120x120x120xf64>
        // expected-remark @below {{reuse-analysis: ref load x1 kinds=[stream,stream,spat] iterFP=[111392,944,8]}}
        %5 = affine.load %A[%i, %j, %k - 1] : memref<120x120x120xf64>
        // expected-remark @below {{reuse-analysis: ref load x1 kinds=[stream,stream,spat] iterFP=[111392,944,8]}}
        %6 = affine.load %A[%i, %j, %k + 1] : memref<120x120x120xf64>
        %s1 = arith.addf %0, %1 : f64
        %s2 = arith.addf %s1, %2 : f64
        %s3 = arith.addf %s2, %3 : f64
        %s4 = arith.addf %s3, %4 : f64
        %s5 = arith.addf %s4, %5 : f64
        %s6 = arith.addf %s5, %6 : f64
        %r = arith.mulf %s6, %c : f64
        // expected-remark @below {{reuse-analysis: ref store x1 kinds=[stream,stream,spat] iterFP=[111392,944,8]}}
        affine.store %r, %B[%i, %j, %k] : memref<120x120x120xf64>
      }
    }
  }
  return
}
