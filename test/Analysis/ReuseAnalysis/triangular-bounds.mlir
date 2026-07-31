// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-test-reuse-analysis{cache-bytes=32768 accept-trip-upper-bounds=true}))' -verify-diagnostics

// Triangular (symbolic) upper bounds.  `getConstantTripCount` fails on
// `affine.for %j = 0 to #map(%i)`, and `hasConstantUpperBound()` is false, so
// before this was supported every one of these bands was refused outright --
// which is what kept syrk / cholesky / symm / syr2k out of the tiler on real
// PolyBench input.  With `accept-trip-upper-bounds` the bound is resolved by
// substituting the enclosing band loop's range, giving a (rectangular) upper
// bound flagged via `tripExact`.
//
// Without the flag the behaviour is unchanged -- see triangular-bounds-off.mlir.

#tri = affine_map<(d0) -> (d0 + 1)>
#diag = affine_map<(d0) -> (d0)>

// syrk shape: j = 0..i, so the real space is N(N+1)/2*M, and the bound
// resolves to the rectangular 1200 (i < 1200 => i + 1 <= 1200).
func.func @syrk(%C: memref<1200x1200xf64>, %A: memref<1200x1000xf64>) {
  // expected-remark @below {{reuse-analysis: band depth=3 trips=[1200, 1000, 1200] footprint=21120000 evictedReuse=[1,1,0] anyTemporal=1 tripExact=[1,1,0]}}
  affine.for %i = 0 to 1200 {
    affine.for %k = 0 to 1000 {
      affine.for %j = 0 to #tri(%i) {
        // expected-remark @below {{reuse-analysis: ref load x1 kinds=[stream,spat,inv] iterFP=[8000,64,64]}}
        %aik = affine.load %A[%i, %k] : memref<1200x1000xf64>
        // expected-remark @below {{reuse-analysis: ref load x1 kinds=[inv,spat,stream] iterFP=[9600000,76800,64]}}
        %ajk = affine.load %A[%j, %k] : memref<1200x1000xf64>
        // expected-remark @below {{reuse-analysis: ref store x2 kinds=[stream,inv,spat] iterFP=[9600,9600,64]}}
        %c = affine.load %C[%i, %j] : memref<1200x1200xf64>
        %m = arith.mulf %aik, %ajk : f64
        %s = arith.addf %c, %m : f64
        affine.store %s, %C[%i, %j] : memref<1200x1200xf64>
      }
    }
  }
  return
}

// cholesky shape: TWO nested triangular bounds, the inner one resolved against
// the middle loop which is itself only upper-bounded.  Note the chaining is
// tighter than a flat rectangular bound would be -- i < 2000 gives j < 1999
// gives k < 1998.
func.func @double_triangular(%A: memref<2000x2000xf64>) {
  // expected-remark @below {{reuse-analysis: band depth=3 trips=[2000, 1999, 1998] footprint=32000000 evictedReuse=[1,0,0] anyTemporal=1 tripExact=[1,0,0]}}
  affine.for %i = 0 to 2000 {
    affine.for %j = 0 to #diag(%i) {
      affine.for %k = 0 to #diag(%j) {
        // expected-remark @below {{reuse-analysis: ref load x1 kinds=[stream,inv,spat] iterFP=[16000,16000,64]}}
        %a = affine.load %A[%i, %k] : memref<2000x2000xf64>
        // expected-remark @below {{reuse-analysis: ref load x1 kinds=[inv,stream,spat] iterFP=[31984000,16000,64]}}
        %b = affine.load %A[%j, %k] : memref<2000x2000xf64>
        // expected-remark @below {{reuse-analysis: ref store x2 kinds=[stream,spat,inv] iterFP=[16000,64,64]}}
        %c = affine.load %A[%i, %j] : memref<2000x2000xf64>
        %m = arith.mulf %a, %b : f64
        %s = arith.subf %c, %m : f64
        affine.store %s, %A[%i, %j] : memref<2000x2000xf64>
      }
    }
  }
  return
}

// A step on the triangular loop must be divided out.  `getConstantTripCount`
// does the ceilDiv for constant bounds; the upper-bound path has to do it
// itself, and forgetting to would report 4x the trips here.
func.func @triangular_step(%A: memref<1024x1024xf64>) {
  // expected-remark @below {{reuse-analysis: band depth=2 trips=[1024, 256] footprint=2097152 evictedReuse=[0,0] anyTemporal=0 tripExact=[1,0]}}
  affine.for %i = 0 to 1024 {
    affine.for %j = 0 to #tri(%i) step 4 {
      // expected-remark @below {{reuse-analysis: ref store x2 kinds=[stream,spat] iterFP=[2048,64]}}
      %a = affine.load %A[%i, %j] : memref<1024x1024xf64>
      %s = arith.mulf %a, %a : f64
      affine.store %s, %A[%i, %j] : memref<1024x1024xf64>
    }
  }
  return
}

// Negative: a genuinely unknown bound.  The trip count depends on a runtime
// symbol, not on an enclosing band IV, so there is no range to substitute and
// the band stays refused even with the flag.
#dyn = affine_map<()[s0] -> (s0)>

func.func @dynamic_bound(%A: memref<64x64xf64>, %n: index) {
  // expected-remark @below {{reuse-analysis: band UNANALYZABLE}}
  affine.for %j = 0 to #dyn()[%n] {
    %a = affine.load %A[%j, %j] : memref<64x64xf64>
    %s = arith.mulf %a, %a : f64
    affine.store %s, %A[%j, %j] : memref<64x64xf64>
  }
  return
}
