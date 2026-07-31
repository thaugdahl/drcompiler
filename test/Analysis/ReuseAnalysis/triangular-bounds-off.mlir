// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-test-reuse-analysis{cache-bytes=32768}))' -verify-diagnostics

// Companion to triangular-bounds.mlir: WITHOUT `accept-trip-upper-bounds` a
// symbolic (triangular) bound is still refused outright.  Clients that need
// exactness -- dr-affine-loop-distribute, which treats a refusal optimistically
// as "reuse benefit present" -- depend on this staying conservative.

#tri = affine_map<(d0) -> (d0 + 1)>

func.func @syrk(%C: memref<1200x1200xf64>, %A: memref<1200x1000xf64>) {
  // expected-remark @below {{reuse-analysis: band UNANALYZABLE}}
  affine.for %i = 0 to 1200 {
    affine.for %k = 0 to 1000 {
      affine.for %j = 0 to #tri(%i) {
        %aik = affine.load %A[%i, %k] : memref<1200x1000xf64>
        %ajk = affine.load %A[%j, %k] : memref<1200x1000xf64>
        %c = affine.load %C[%i, %j] : memref<1200x1200xf64>
        %m = arith.mulf %aik, %ajk : f64
        %s = arith.addf %c, %m : f64
        affine.store %s, %C[%i, %j] : memref<1200x1200xf64>
      }
    }
  }
  return
}
