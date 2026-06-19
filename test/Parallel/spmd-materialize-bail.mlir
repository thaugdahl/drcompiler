// RUN: dr-opt %s -split-input-file --pass-pipeline='builtin.module(dr-par-bubbles{par-spmd})' -verify-diagnostics

// S2 soundness bails (PARALLEL_SPMD_SPEC.md §7 / §11.5 blockers): the
// materializer refuses to mutate unless every precondition holds, emitting a
// remark (on the function) that names the reason.  No par.region is produced.

// A dynamic shard extent cannot be a constant par.forall.
// expected-remark @below {{par-spmd: not materialized (dynamic shard extent)}}
func.func @bail_dyn(%n : index) {
  %A = memref.alloc(%n) : memref<?xf32>
  %B = memref.alloc(%n) : memref<?xf32>
  affine.for %i = 0 to %n {
    %a = affine.load %A[%i] : memref<?xf32>
    affine.store %a, %B[%i] : memref<?xf32>
  }
  return
}

// -----

// An off-axis band (the third loop iterates a different extent, so it is not on
// the chosen shard axis) breaks whole-function widening.
// expected-remark @below {{par-spmd: not materialized (off-axis or non-materializable band)}}
func.func @bail_offaxis() {
  %z = arith.constant 0.0 : f32
  %A = memref.alloc() : memref<64xf32>
  %B = memref.alloc() : memref<32xf32>
  affine.for %i = 0 to 64 {
    affine.store %z, %A[%i] : memref<64xf32>
  }
  affine.for %i = 0 to 64 {
    affine.store %z, %A[%i] : memref<64xf32>
  }
  affine.for %i = 0 to 32 {
    affine.store %z, %B[%i] : memref<32xf32>
  }
  return
}

// -----

// A non-band op (an alloc feeding a later band) sits between two shard bands:
// moving the second band ahead of it would reorder, so bail.
// expected-remark @below {{par-spmd: not materialized (op between shard bands)}}
func.func @bail_interleaved() {
  %z = arith.constant 0.0 : f32
  %A = memref.alloc() : memref<64xf32>
  affine.for %i = 0 to 64 {
    affine.store %z, %A[%i] : memref<64xf32>
  }
  %B = memref.alloc() : memref<64xf32>
  affine.for %i = 0 to 64 {
    %a = affine.load %A[%i] : memref<64xf32>
    affine.store %a, %B[%i] : memref<64xf32>
  }
  return
}
