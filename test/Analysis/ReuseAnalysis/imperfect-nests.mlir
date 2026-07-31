// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-test-reuse-analysis{cache-bytes=32768 all-bands=true accept-trip-upper-bounds=true}))' -verify-diagnostics

// Bands that do not start at the top of the nest.
//
// Top-level-only enumeration plus a band-IVs-only subscript model rejected
// every one of these shapes.  Over PolyBench-L that was ALL 26 rejected
// bands, leaving 15 of 30 kernels with no analyzable band whatsoever.  Three
// separate restrictions had to go:
//
//   1. band enumeration must descend through an imperfect loop (all-bands),
//   2. an ENCLOSING loop's IV is fixed for the band, so it may appear in a
//      subscript and contributes no extent (RefInfo::outerCoeff),
//   3. a loop BELOW the band sweeps its whole range each band iteration, so
//      it may appear in a subscript and contributes a constant extent
//      (RefInfo::innerExtent).
//
// With all-bands every maximal band is reported, so a reference inside a
// nested band draws one remark per band containing it -- hence the stacked
// @+N expectations.  Bands are emitted innermost-first (Operation::walk is
// post-order), and within a band: band line, then groups, then refs.

// (1) + (3): the ping-pong stencil.  `getPerfectlyNestedLoops` truncates at
// %t, so without all-bands the two [%i, %j] bands are invisible AND the [%t]
// band is rejected -- nothing is analyzed at all.
//
// Every footprint here is exact.  In the [%i, %j] bands the three %A
// references hull-union over rows [-1, 63) = 64 rows; each row spans 62 f64 =
// 496 B, rounded up to one 512 B line multiple, so %A costs 64*512 = 32768 B
// and the single %B row set costs 62*512 = 31744 B (64512 together).  The
// second band touches one 62-row set of each array: 63488 B.
//
// In the [%t] band both %i and %j sit BELOW the band, so each contributes its
// full 62-iteration sweep as innerExtent: every reference is invariant in %t
// with iterFP 62*512 = 31744, and the band footprint is the same 64512 B.
// 64512 > 32768 makes that reuse correctly EVICTED -- the one verdict here
// that asks for a transform, and one the old model could not reach at all.
func.func @pingpong(%A: memref<64x64xf64>, %B: memref<64x64xf64>) {
  // expected-remark @below {{reuse-analysis: band depth=1 trips=[100] footprint=64512 evictedReuse=[1] anyTemporal=1}}
  affine.for %t = 0 to 100 {
    // expected-remark @below {{reuse-analysis: band depth=2 trips=[62, 62] footprint=64512 evictedReuse=[0,0] anyTemporal=0}}
    affine.for %i = 1 to 63 {
      affine.for %j = 1 to 63 {
        // expected-remark @+4 {{reuse-analysis: group members=3 span=[2, 0] carries=[1,0]}}
        // expected-remark @+3 {{reuse-analysis: ref load x1 kinds=[stream,spat] iterFP=[512,64]}}
        // expected-remark @+2 {{reuse-analysis: group members=4 span=[2, 0] carries=[0]}}
        // expected-remark @+1 {{reuse-analysis: ref load x1 kinds=[inv] iterFP=[31744]}}
        %0 = affine.load %A[%i - 1, %j] : memref<64x64xf64>
        // expected-remark @+2 {{reuse-analysis: ref load x1 kinds=[stream,spat] iterFP=[512,64]}}
        // expected-remark @+1 {{reuse-analysis: ref load x1 kinds=[inv] iterFP=[31744]}}
        %1 = affine.load %A[%i, %j] : memref<64x64xf64>
        // expected-remark @+2 {{reuse-analysis: ref load x1 kinds=[stream,spat] iterFP=[512,64]}}
        // expected-remark @+1 {{reuse-analysis: ref load x1 kinds=[inv] iterFP=[31744]}}
        %2 = affine.load %A[%i + 1, %j] : memref<64x64xf64>
        %3 = arith.addf %0, %1 : f64
        %4 = arith.addf %3, %2 : f64
        // expected-remark @+3 {{reuse-analysis: ref store x1 kinds=[stream,spat] iterFP=[512,64]}}
        // expected-remark @+2 {{reuse-analysis: group members=2 span=[0, 0] carries=[0]}}
        // expected-remark @+1 {{reuse-analysis: ref store x1 kinds=[inv] iterFP=[31744]}}
        affine.store %4, %B[%i, %j] : memref<64x64xf64>
      }
    }
    // expected-remark @below {{reuse-analysis: band depth=2 trips=[62, 62] footprint=63488 evictedReuse=[0,0] anyTemporal=0}}
    affine.for %i = 1 to 63 {
      affine.for %j = 1 to 63 {
        // expected-remark @+2 {{reuse-analysis: ref load x1 kinds=[stream,spat] iterFP=[512,64]}}
        // expected-remark @+1 {{reuse-analysis: ref load x1 kinds=[inv] iterFP=[31744]}}
        %0 = affine.load %B[%i, %j] : memref<64x64xf64>
        // expected-remark @+2 {{reuse-analysis: ref store x1 kinds=[stream,spat] iterFP=[512,64]}}
        // expected-remark @+1 {{reuse-analysis: ref store x1 kinds=[inv] iterFP=[31744]}}
        affine.store %0, %A[%i, %j] : memref<64x64xf64>
      }
    }
  }
  return
}

// (2): the inner [%j] band's subscripts name the ENCLOSING %i.  %i is fixed
// for the whole band, so it widens nothing: the band spans one 512xf64 row of
// %A (512*8 = 4096 B) plus the one %B element's 64 B line = 4160 B, and %B[%i]
// is correctly invariant in %j.  The three syntactic accesses to %B[%i] dedup
// to one entry -- x2 within the [%j] band, x3 within the [%i] band -- reported
// as a store because a store folded into it.
//
// anyTemporal is 0 despite that invariance: %B[%i] is a single element, so its
// per-iteration footprint is exactly one 64 B line and the reuse is degenerate
// (an accumulator that lives in a register, which tiling cannot improve).  The
// gate requires strictly more than a line.
func.func @encl_iv(%A: memref<512x512xf64>, %B: memref<512xf64>) {
  %init = arith.constant 0.0 : f64
  // expected-remark @below {{reuse-analysis: band depth=1 trips=[512] footprint=2101248 evictedReuse=[0] anyTemporal=0}}
  affine.for %i = 0 to 512 {
    // expected-remark @below {{reuse-analysis: ref store x3 kinds=[spat] iterFP=[64]}}
    affine.store %init, %B[%i] : memref<512xf64>
    // expected-remark @below {{reuse-analysis: band depth=1 trips=[512] footprint=4160 evictedReuse=[0] anyTemporal=0}}
    affine.for %j = 0 to 512 {
      // expected-remark @+2 {{reuse-analysis: ref load x1 kinds=[spat] iterFP=[64]}}
      // expected-remark @+1 {{reuse-analysis: ref load x1 kinds=[stream] iterFP=[4096]}}
      %0 = affine.load %A[%i, %j] : memref<512x512xf64>
      // expected-remark @below {{reuse-analysis: ref store x2 kinds=[inv] iterFP=[64]}}
      %1 = affine.load %B[%i] : memref<512xf64>
      %2 = arith.addf %0, %1 : f64
      affine.store %2, %B[%i] : memref<512xf64>
    }
  }
  return
}

// (3): the symm shape -- an [%i, %j] band whose body holds a deeper %k
// reduction loop.  %k sweeps all 512 values for every (%i, %j), so %C[%k, %j]
// spans 512 rows even though %k is not a band IV.  That makes %C invariant in
// %i (the same 2 MiB column strip is re-read for every %i) and the reuse
// distance 2105344 B, far past the 32 KiB cache -- exactly the reuse a tiler
// must see, and exactly what the old model could not represent.
//
// Band footprint 6291456 = 3 arrays x 512 rows x 4096 B: %C[%k, %j] spans
// 1+511 rows by 512 cols, %A[%i, %k] 512 rows by 1+511 cols, %B[%i, %j] 512
// by 512.
func.func @inner_reduction(%A: memref<512x512xf64>, %B: memref<512x512xf64>,
                           %C: memref<512x512xf64>) {
  %zero = arith.constant 0.0 : f64
  // expected-remark @below {{reuse-analysis: band depth=2 trips=[512, 512] footprint=6291456 evictedReuse=[1,1] anyTemporal=1}}
  affine.for %i = 0 to 512 {
    affine.for %j = 0 to 512 {
      // expected-remark @below {{reuse-analysis: ref store x3 kinds=[stream,spat] iterFP=[4096,64]}}
      affine.store %zero, %B[%i, %j] : memref<512x512xf64>
      // expected-remark @below {{reuse-analysis: band depth=1 trips=[512] footprint=36928 evictedReuse=[0] anyTemporal=0}}
      affine.for %k = 0 to 512 {
        // expected-remark @+2 {{reuse-analysis: ref load x1 kinds=[stream] iterFP=[64]}}
        // expected-remark @+1 {{reuse-analysis: ref load x1 kinds=[inv,spat] iterFP=[2097152,32768]}}
        %0 = affine.load %C[%k, %j] : memref<512x512xf64>
        // expected-remark @+2 {{reuse-analysis: ref load x1 kinds=[spat] iterFP=[64]}}
        // expected-remark @+1 {{reuse-analysis: ref load x1 kinds=[stream,inv] iterFP=[4096,4096]}}
        %1 = affine.load %A[%i, %k] : memref<512x512xf64>
        %2 = arith.mulf %0, %1 : f64
        // expected-remark @below {{reuse-analysis: ref store x2 kinds=[inv] iterFP=[64]}}
        %3 = affine.load %B[%i, %j] : memref<512x512xf64>
        %4 = arith.addf %3, %2 : f64
        affine.store %4, %B[%i, %j] : memref<512x512xf64>
      }
    }
  }
  return
}
