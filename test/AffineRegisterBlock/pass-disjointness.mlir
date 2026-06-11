// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-affine-stencil-time-tile{force-tile=true tile-t=4 tile-s=8}))' | FileCheck %s --check-prefix=STENCIL
// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block{mr=8 nr=16 vl=16}))' | FileCheck %s --check-prefix=REGBLOCK

// WP6 (COSTMODEL_V4_SPEC §8): the stencil-time-tile and register-block passes
// are DISJOINT matchers, so they compose safely in the single drcomp-v4
// pipeline.  The stencil pass needs a time loop over stencil nests (no
// k-invariant accumulator); register-block needs a reduction band (no time
// loop).  Neither touches the other's kernel.

// The stencil pass is a no-op on the GEMM reduction band (no skew windows).
// STENCIL-LABEL: func.func @gemm
// STENCIL-NOT: #map
// STENCIL-LABEL: func.func @jacobi
func.func @gemm(%A: memref<64x64xf64>, %B: memref<64x64xf64>, %C: memref<64x64xf64>) {
  affine.for %i = 0 to 64 { affine.for %j = 0 to 64 { affine.for %k = 0 to 64 {
    %a = affine.load %A[%i, %k] : memref<64x64xf64>
    %b = affine.load %B[%k, %j] : memref<64x64xf64>
    %c = affine.load %C[%i, %j] : memref<64x64xf64>
    %p = arith.mulf %a, %b : f64
    %s = arith.addf %c, %p : f64
    affine.store %s, %C[%i, %j] : memref<64x64xf64>
  }}}
  return
}

// Register-block is a no-op on the jacobi-2d ping-pong stencil (no reduction
// accumulator -> no iter_args / vector micro-kernel).
// REGBLOCK-LABEL: func.func @gemm
// REGBLOCK-LABEL: func.func @jacobi
// REGBLOCK-NOT: iter_args
// REGBLOCK-NOT: vector.broadcast
func.func @jacobi(%A: memref<64x64xf64>, %B: memref<64x64xf64>) {
  %c = arith.constant 0.2 : f64
  affine.for %t = 0 to 100 {
    affine.for %i = 1 to 63 { affine.for %j = 1 to 63 {
      %0 = affine.load %A[%i, %j] : memref<64x64xf64>
      %1 = affine.load %A[%i - 1, %j] : memref<64x64xf64>
      %2 = arith.addf %0, %1 : f64
      %3 = arith.mulf %2, %c : f64
      affine.store %3, %B[%i, %j] : memref<64x64xf64>
    }}
    affine.for %i = 1 to 63 { affine.for %j = 1 to 63 {
      %0 = affine.load %B[%i, %j] : memref<64x64xf64>
      %1 = affine.load %B[%i - 1, %j] : memref<64x64xf64>
      %2 = arith.addf %0, %1 : f64
      %3 = arith.mulf %2, %c : f64
      affine.store %3, %A[%i, %j] : memref<64x64xf64>
    }}
  }
  return
}
