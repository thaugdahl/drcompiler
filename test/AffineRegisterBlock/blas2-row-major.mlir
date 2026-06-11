// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block{mr=8 nr=16 vl=16}))' | FileCheck %s

// WP5 (COSTMODEL_V4_SPEC §6 — mechanism corrected by spike: row-major loop
// interchange, NOT the spec's block-interleave).  A BLAS-2 column-major
// streaming nest `for j { for i: ... A[i][j] ... }` (i inner -> A strided by a
// full row) under a SEQUENTIAL outer sweep is interchanged to `for i { for j }`
// so A is stride-1 (row-major).  Validated on gramschmidt: 1.56x -> 8.9x XL,
// SINK bit-identical.  A PARALLEL sweep (GEMM) must keep the reduction innermost
// for register-blocking and must NOT be interchanged.

// @blas2: sweep k is sequential (the second nest writes A, read by the next k's
// first nest).  Both nests are column-major (A[i][j], i inner) and interchange
// to i-outer / j-inner; the load then indexes A as [middle-IV, inner-IV].
// CHECK-LABEL: func.func @blas2
// CHECK: affine.for %[[K:.*]] = 0 to 512 {
// CHECK:   affine.for %[[I:.*]] = 0 to 512 {
// CHECK:     affine.for %[[J:.*]] = 0 to 512 {
// CHECK:       affine.load %{{.*}}[%[[I]], %[[J]]] : memref<512x512xf64>
func.func @blas2(%A: memref<512x512xf64>, %Q: memref<512x512xf64>, %R: memref<512x512xf64>) {
  affine.for %k = 0 to 512 {
    affine.for %j = 0 to 512 {
      affine.for %i = 0 to 512 {
        %a = affine.load %A[%i, %j] : memref<512x512xf64>
        %q = affine.load %Q[%i, %k] : memref<512x512xf64>
        %p = arith.mulf %q, %a : f64
        %r = affine.load %R[%k, %j] : memref<512x512xf64>
        %s = arith.addf %r, %p : f64
        affine.store %s, %R[%k, %j] : memref<512x512xf64>
      }
    }
    affine.for %j = 0 to 512 {
      affine.for %i = 0 to 512 {
        %a = affine.load %A[%i, %j] : memref<512x512xf64>
        %q = affine.load %Q[%i, %k] : memref<512x512xf64>
        %r = affine.load %R[%k, %j] : memref<512x512xf64>
        %p = arith.mulf %q, %r : f64
        %d = arith.subf %a, %p : f64
        affine.store %d, %A[%i, %j] : memref<512x512xf64>
      }
    }
  }
  return
}

// @gemm: sweep i is PARALLEL -> NOT interchanged; register-blocked instead
// (unroll-jammed outer loops + an iter_args reduction over k).
// CHECK-LABEL: func.func @gemm
// CHECK: affine.for %{{.*}} = 0 to 512 step 8
// CHECK: iter_args
func.func @gemm(%A: memref<512x512xf64>, %B: memref<512x512xf64>, %C: memref<512x512xf64>) {
  affine.for %i = 0 to 512 {
    affine.for %j = 0 to 512 {
      affine.for %k = 0 to 512 {
        %a = affine.load %A[%i, %k] : memref<512x512xf64>
        %b = affine.load %B[%k, %j] : memref<512x512xf64>
        %c = affine.load %C[%i, %j] : memref<512x512xf64>
        %p = arith.mulf %a, %b : f64
        %s = arith.addf %c, %p : f64
        affine.store %s, %C[%i, %j] : memref<512x512xf64>
      }
    }
  }
  return
}
