// RUN: dr-opt %s -split-input-file --pass-pipeline='builtin.module(dr-par-bubbles{par-test-spmd})' -verify-diagnostics

// S0/S1 (PARALLEL_SPMD_SPEC.md): pick a whole-function shard axis and classify
// each inter-band edge for barrier elision. Diagnostic-only.

// Owner-aligned eltwise chain over one axis -> every edge ELIDEs.
// expected-remark @below {{par-shard: extent=64 bands=3/3}}
// expected-remark @below {{par-spmd: elide=2 halo=0 redistribute=0 barrier=0 (of 2 shard edges)}}
func.func @spmd_elide() {
  %z = arith.constant 0.0 : f32
  %A = memref.alloc() : memref<64xf32>
  %B = memref.alloc() : memref<64xf32>
  %C = memref.alloc() : memref<64xf32>
  affine.for %i = 0 to 64 {
    %a = affine.load %A[%i] : memref<64xf32>
    %e = math.exp %a : f32
    affine.store %e, %B[%i] : memref<64xf32>
  }
  // expected-remark @below {{par-edge: ELIDE}}
  affine.for %i = 0 to 64 {
    %b = affine.load %B[%i] : memref<64xf32>
    %t = math.tanh %b : f32
    affine.store %t, %C[%i] : memref<64xf32>
  }
  // expected-remark @below {{par-edge: ELIDE}}
  affine.for %i = 0 to 64 {
    %c = affine.load %C[%i] : memref<64xf32>
    %s = arith.addf %c, %z : f32
    affine.store %s, %A[%i] : memref<64xf32>
  }
  return
}

// -----

// Constant-offset producer->consumer on one buffer -> HALO.
// expected-remark @below {{par-shard: extent=100 bands=2/2}}
// expected-remark @below {{par-spmd: elide=0 halo=1 redistribute=0 barrier=0 (of 1 shard edges)}}
func.func @spmd_halo() {
  %z = arith.constant 0.0 : f32
  %T = memref.alloc() : memref<128xf32>
  %O = memref.alloc() : memref<128xf32>
  affine.for %i = 0 to 100 { affine.store %z, %T[%i] : memref<128xf32> }
  // expected-remark @below {{par-edge: HALO}}
  affine.for %i = 0 to 100 {
    %t = affine.load %T[%i + 1] : memref<128xf32>
    affine.store %t, %O[%i] : memref<128xf32>
  }
  return
}

// -----

// Reverse read (IV coefficient flips) -> REDISTRIBUTE.
// expected-remark @below {{par-shard: extent=128 bands=2/2}}
// expected-remark @below {{par-spmd: elide=0 halo=0 redistribute=1 barrier=0 (of 1 shard edges)}}
func.func @spmd_redistribute() {
  %z = arith.constant 0.0 : f32
  %M = memref.alloc() : memref<128xf32>
  %O = memref.alloc() : memref<128xf32>
  affine.for %i = 0 to 128 { affine.store %z, %M[%i] : memref<128xf32> }
  // expected-remark @below {{par-edge: REDISTRIBUTE}}
  affine.for %i = 0 to 128 {
    %v = affine.load %M[127 - %i] : memref<128xf32>
    affine.store %v, %O[%i] : memref<128xf32>
  }
  return
}
