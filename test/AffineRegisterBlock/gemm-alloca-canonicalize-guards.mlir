// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-scalar-reduction-demote,affine-register-block{mr=8 nr=16 cpu-cost-model-file=%S/Inputs/gemm-model.json},dr-scalar-reduction-promote))' | FileCheck %s

// WP-T5c legality guards (adversarial review): canonicalizeAllocaGemm must REFUSE
// to promote+fission patterns where the fission would break a dependence, even
// under a GEMM model.  In each case the scalar alloca must SURVIVE untouched (the
// nest is left correct-but-unoptimized).  A legit GEMM never reads its own output
// C in the j-loop; any C read (in-place C==A/B, beta/residual C=acc+C, fused
// reuse) is the rejection signal.

// (1) In-place: output aliases multiplicand A (C == A).  The INIT nest would zero
//     A before the GEMM nest reads it -> reject.
// CHECK-LABEL: func @inplace_C_is_A
// CHECK: memref.alloca
func.func @inplace_C_is_A(%A: memref<8x32xf32>, %B: memref<32x8xf32>, %bias: memref<8xf32>) {
  %z = arith.constant 0.0 : f32
  affine.for %i = 0 to 8 { affine.for %j = 0 to 8 {
    %a = memref.alloca() : memref<f32>
    affine.store %z, %a[] : memref<f32>
    affine.for %k = 0 to 32 {
      %x = affine.load %A[%i,%k] : memref<8x32xf32>
      %y = affine.load %B[%k,%j] : memref<32x8xf32>
      %p = arith.mulf %x, %y : f32
      %acc = affine.load %a[] : memref<f32>
      %s = arith.addf %p, %acc : f32
      affine.store %s, %a[] : memref<f32> }
    %f = affine.load %a[] : memref<f32>
    %bb = affine.load %bias[%j] : memref<8xf32>
    %r = arith.addf %f, %bb : f32
    affine.store %r, %A[%i,%j] : memref<8x32xf32> } }
  return
}

// (2) Beta / residual accumulate: epilogue reads the output C[i,j] (C = acc + C).
//     Fission breaks the WAR dependence (EPI reads C before GEMM writes it) ->
//     reject.
// CHECK-LABEL: func @beta_reads_output
// CHECK: memref.alloca
func.func @beta_reads_output(%A: memref<8x32xf32>, %B: memref<32x8xf32>, %C: memref<8x8xf32>) {
  %z = arith.constant 0.0 : f32
  affine.for %i = 0 to 8 { affine.for %j = 0 to 8 {
    %a = memref.alloca() : memref<f32>
    affine.store %z, %a[] : memref<f32>
    affine.for %k = 0 to 32 {
      %x = affine.load %A[%i,%k] : memref<8x32xf32>
      %y = affine.load %B[%k,%j] : memref<32x8xf32>
      %p = arith.mulf %x, %y : f32
      %acc = affine.load %a[] : memref<f32>
      %s = arith.addf %p, %acc : f32
      affine.store %s, %a[] : memref<f32> }
    %f = affine.load %a[] : memref<f32>
    %old = affine.load %C[%i,%j] : memref<8x8xf32>
    %r = arith.addf %f, %old : f32
    affine.store %r, %C[%i,%j] : memref<8x8xf32> } }
  return
}
