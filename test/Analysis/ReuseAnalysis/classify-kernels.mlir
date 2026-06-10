// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-test-reuse-analysis{cache-bytes=32768}))' -verify-diagnostics

// Per-reference reuse classification on the three canonical shapes the
// tiler's profitability gate must distinguish:
//  - matmul: i carries B's temporal reuse (2 MiB stream per iteration —
//    evicted at 32 KiB => tile around i).  k carries C-row reuse, but at
//    512x512 one k-iteration touches only ~8 KiB (B row + C row), which
//    FITS the cache: k is correctly NOT flagged (it flips to 1 at larger
//    rows, e.g. PolyBench EXTRALARGE).  j's only temporal reuse (A[i][k])
//    is a single element — register-degenerate, so j never counts.
//  - streaming matvec accumulation: x's reuse fits the cache (no eviction),
//    everything else streams => not worth tiling.
//  - init nest (no loads): no temporal reuse at all => not worth tiling.

// 512x512xf64 matrices: one row = 4 KiB; B = 2 MiB >> 32 KiB cache.
func.func @matmul(%A: memref<512x512xf64>, %B: memref<512x512xf64>, %C: memref<512x512xf64>) {
  // expected-remark @below {{reuse-analysis: band depth=3 trips=[512, 512, 512] footprint=6291456 evictedReuse=[1,0,0] anyTemporal=1}}
  affine.for %i = 0 to 512 {
    affine.for %k = 0 to 512 {
      affine.for %j = 0 to 512 {
        // expected-remark @below {{reuse-analysis: ref load x1 kinds=[stream,spat,inv] iterFP=[4096,8,8]}}
        %a = affine.load %A[%i, %k] : memref<512x512xf64>
        // expected-remark @below {{reuse-analysis: ref load x1 kinds=[inv,stream,spat] iterFP=[2097152,4096,8]}}
        %b = affine.load %B[%k, %j] : memref<512x512xf64>
        // expected-remark @below {{reuse-analysis: ref store x2 kinds=[stream,inv,spat] iterFP=[4096,4096,8]}}
        %c = affine.load %C[%i, %j] : memref<512x512xf64>
        %mul = arith.mulf %a, %b : f64
        %add = arith.addf %c, %mul : f64
        affine.store %add, %C[%i, %j] : memref<512x512xf64>
      }
    }
  }
  return
}

// y[i] += A[i][j] * x[j]: x (4 KiB) is invariant in i but its reuse distance
// (one full j sweep ~ 8 KiB) fits the 32 KiB cache => no evicted reuse, and
// y[i] / the A row walk gain nothing from tiling.
func.func @matvec(%A: memref<512x512xf64>, %x: memref<512xf64>, %y: memref<512xf64>) {
  // expected-remark @below {{reuse-analysis: band depth=2 trips=[512, 512] footprint=2105344 evictedReuse=[0,0] anyTemporal=1}}
  affine.for %i = 0 to 512 {
    affine.for %j = 0 to 512 {
      // expected-remark @below {{reuse-analysis: ref load x1 kinds=[stream,spat] iterFP=[4096,8]}}
      %a = affine.load %A[%i, %j] : memref<512x512xf64>
      // expected-remark @below {{reuse-analysis: ref load x1 kinds=[inv,spat] iterFP=[4096,8]}}
      %xv = affine.load %x[%j] : memref<512xf64>
      // expected-remark @below {{reuse-analysis: ref store x2 kinds=[spat,inv] iterFP=[8,8]}}
      %yv = affine.load %y[%i] : memref<512xf64>
      %mul = arith.mulf %a, %xv : f64
      %add = arith.addf %yv, %mul : f64
      affine.store %add, %y[%i] : memref<512xf64>
    }
  }
  return
}

// Init nest: store only, no temporal reuse anywhere.
func.func @init(%A: memref<512x512xf64>, %v: f64) {
  // expected-remark @below {{reuse-analysis: band depth=2 trips=[512, 512] footprint=2097152 evictedReuse=[0,0] anyTemporal=0}}
  affine.for %i = 0 to 512 {
    affine.for %j = 0 to 512 {
      // expected-remark @below {{reuse-analysis: ref store x1 kinds=[stream,spat] iterFP=[4096,8]}}
      affine.store %v, %A[%i, %j] : memref<512x512xf64>
    }
  }
  return
}

// Time-loop band whose references use non-band IVs (imperfect below the
// band) is out of the constant-coefficient model: must report UNANALYZABLE,
// never a wrong classification.
func.func @timeloop(%A: memref<512xf64>, %B: memref<512xf64>) {
  // expected-remark @below {{reuse-analysis: band UNANALYZABLE}}
  affine.for %t = 0 to 100 {
    affine.for %i = 1 to 511 {
      %l = affine.load %A[%i - 1] : memref<512xf64>
      affine.store %l, %B[%i] : memref<512xf64>
    }
    affine.for %i = 1 to 511 {
      %l = affine.load %B[%i] : memref<512xf64>
      affine.store %l, %A[%i] : memref<512xf64>
    }
  }
  return
}
