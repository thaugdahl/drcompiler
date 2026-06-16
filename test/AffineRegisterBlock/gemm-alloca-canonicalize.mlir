// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-scalar-reduction-demote,affine-register-block{mr=8 nr=16 cpu-cost-model-file=%S/Inputs/gemm-model.json},dr-scalar-reduction-promote))' | FileCheck %s --check-prefix=CANON
// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-scalar-reduction-demote,affine-register-block{mr=8 nr=16},dr-scalar-reduction-promote))' | FileCheck %s --check-prefix=FLAT

// WP-T5c (TRANSFORMER_KRNL_SPEC): onnx-mlir lowers each Gemm to a scalar-alloca
// accumulator with an inline bias epilogue:
//   for i { for j { %a=alloca; %a=0; for k {%a+=A[i,k]*B[k,j]}; C[i,j]=%a+bias[j] } }
// The rank-0 (scalar) accumulator + imperfect j-body defeat register-block's
// vectorizer (enclosingSpatial needs a spatially-indexed acc) AND the cache
// tiler, so the FFN runs as a scalar memory-round-trip-per-k dependent chain.
// Under a GEMM model (here arch.fma_units => hasExplicitGemmModel),
// canonicalizeAllocaGemm promotes the alloca to the spatial output C[i,j] and
// fissions init / k-reduction / bias into perfect nests, so the existing
// vectorizer crushes the GEMM band (vector.broadcast micro-kernel) -- the
// openai-gpt FFN lever (0.98x -> beats onnx-mlir --O3).  With NO GEMM model the
// transform is inert: the scalar alloca survives and nothing vectorizes
// (byte-identical to pre-T5c).

func.func @ffn(%A: memref<8x32xf32>, %B: memref<32x64xf32>, %bias: memref<64xf32>, %C: memref<8x64xf32>) {
  %z = arith.constant 0.0 : f32
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 64 {
      %a = memref.alloca() : memref<f32>
      affine.store %z, %a[] : memref<f32>
      affine.for %k = 0 to 32 {
        %x = affine.load %A[%i,%k] : memref<8x32xf32>
        %y = affine.load %B[%k,%j] : memref<32x64xf32>
        %p = arith.mulf %x, %y : f32
        %acc = affine.load %a[] : memref<f32>
        %s = arith.addf %p, %acc : f32
        affine.store %s, %a[] : memref<f32>
      }
      %f = affine.load %a[] : memref<f32>
      %bb = affine.load %bias[%j] : memref<64xf32>
      %r = arith.addf %f, %bb : f32
      affine.store %r, %C[%i,%j] : memref<8x64xf32>
    }
  }
  return
}

// Canonicalized: the scalar accumulator is promoted away and the perfect GEMM
// band vectorizes into a broadcast micro-kernel.
// CANON: vector.broadcast
// CANON-NOT: memref.alloca

// Default machine: no GEMM model => transform inert => scalar alloca survives,
// no vectorization.
// FLAT: memref.alloca
// FLAT-NOT: vector.broadcast
