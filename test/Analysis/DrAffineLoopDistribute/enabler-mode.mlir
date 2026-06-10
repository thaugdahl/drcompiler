// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-affine-loop-distribute{mode=enabler emit-rationale=true}))' -verify-diagnostics
// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-affine-loop-distribute{mode=always}))' | FileCheck %s --check-prefix=ALWAYS

// Legal split that deepens no perfect band: %v is computed at i-body level
// and consumed inside BOTH j nests, so every copy keeps a live body-level op
// and stays imperfect.  'enabler' must skip (the tiler could not analyze
// either copy, so fission only costs loop overhead); 'always' still splits.

// ALWAYS-LABEL: func.func @shared_scalar
// ALWAYS-COUNT-4: affine.for
func.func @shared_scalar(%A: memref<64x64xf64>, %B: memref<64x64xf64>) {
  // expected-remark @below {{distribute-rationale: SKIP legal split (no band deepened)}}
  affine.for %i = 0 to 64 {
    %ic = arith.index_cast %i : index to i64
    %v = arith.sitofp %ic : i64 to f64
    affine.for %j = 0 to 64 {
      affine.store %v, %A[%i, %j] : memref<64x64xf64>
    }
    affine.for %j = 0 to 64 {
      affine.store %v, %B[%i, %j] : memref<64x64xf64>
    }
  }
  return
}

// Same shape but the scalar feeds only the FIRST nest: the second copy
// erases it (dead) and becomes a clean deeper band — enabler splits.

func.func @single_consumer(%A: memref<64x64xf64>, %B: memref<64x64xf64>, %C: memref<64x64xf64>) {
  // expected-remark @below {{distribute-rationale: SPLIT into 2 loops}}
  affine.for %i = 0 to 64 {
    %ic = arith.index_cast %i : index to i64
    %v = arith.sitofp %ic : i64 to f64
    affine.for %j = 0 to 64 {
      affine.store %v, %A[%i, %j] : memref<64x64xf64>
    }
    affine.for %j = 0 to 64 {
      %b = affine.load %B[%i, %j] : memref<64x64xf64>
      affine.store %b, %C[%i, %j] : memref<64x64xf64>
    }
  }
  return
}

// Legal, band-deepening split whose units SHARE a read array with no
// downstream reuse benefit (atax shape: both j-nests sweep row A[i][:], and
// nothing in the post-split bands carries evicted temporal reuse).  The
// fused form gets the row reuse from the cache for free; splitting refetches
// it a full sweep later — measured distribute-regblock 0.76x on atax at
// EXTRALARGE.  The locality guard must skip.

func.func @shared_row(%A: memref<64x64xf64>, %t: memref<64xf64>, %y: memref<64xf64>) {
  // expected-remark @below {{distribute-rationale: SKIP legal split (shared data, no reuse benefit)}}
  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 64 {
      %a = affine.load %A[%i, %j] : memref<64x64xf64>
      %tv = affine.load %t[%i] : memref<64xf64>
      %m = arith.mulf %a, %tv : f64
      affine.store %m, %t[%i] : memref<64xf64>
    }
    affine.for %j = 0 to 64 {
      %a = affine.load %A[%i, %j] : memref<64x64xf64>
      %yv = affine.load %y[%j] : memref<64xf64>
      %s = arith.addf %yv, %a : f64
      affine.store %s, %y[%j] : memref<64xf64>
    }
  }
  return
}
