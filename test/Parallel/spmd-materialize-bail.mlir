// RUN: dr-opt %s -split-input-file --pass-pipeline='builtin.module(dr-par-bubbles{par-spmd})' -verify-diagnostics

// S2 soundness bails (PARALLEL_SPMD_SPEC.md §7 / §11.5 blockers): the
// materializer refuses to mutate unless every precondition holds, emitting a
// remark (on the function) that names the reason.  No par.region is produced.

// A complex (non-bare) affine upper bound is not a simple runtime extent: bail.
// (A bare `0 to %n` DOES materialize -- see spmd-materialize.mlir @dyn_batch.)
// expected-remark @below {{par-spmd: not materialized (non-constant lb / complex shard bound)}}
func.func @bail_complexbound(%n : index) {
  %A = memref.alloc(%n) : memref<?xf32>
  %B = memref.alloc(%n) : memref<?xf32>
  affine.for %i = 0 to affine_map<()[s0] -> (s0 + 1)>()[%n] {
    %a = affine.load %A[%i] : memref<?xf32>
    affine.store %a, %B[%i] : memref<?xf32>
  }
  return
}

// -----

// Two bands share the shard-axis MAP (`0 to %dyn`) but use DIFFERENT runtime
// extents (%n vs %m): owner-computes needs ONE shard space, so the second band
// is off-axis -> bail.  (This is the cross-layer dim-equality limit: real ONNX
// per-layer batch dims are distinct SSA values not proven equal.)
// expected-remark @below {{par-spmd: not materialized (off-axis or non-materializable band)}}
func.func @bail_diffdyn(%n : index, %m : index) {
  %z = arith.constant 0.0 : f32
  %A = memref.alloc(%n) : memref<?xf32>
  %C = memref.alloc(%m) : memref<?xf32>
  affine.for %i = 0 to %n {
    affine.store %z, %A[%i] : memref<?xf32>
  }
  affine.for %i = 0 to %m {
    affine.store %z, %C[%i] : memref<?xf32>
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

// A non-hoistable op (a memref.load reading the first band's output) sits
// between two shard bands: it cannot be hoisted above band 1 (it reads what
// band 1 writes), and moving band 2 past it would reorder -> bail.  (Scratch
// allocs / pure index ops between bands DO hoist -- see spmd-materialize.mlir.)
// expected-remark @below {{par-spmd: not materialized (non-hoistable op between shard bands)}}
func.func @bail_nonhoistable() {
  %c0 = arith.constant 0 : index
  %z = arith.constant 0.0 : f32
  %A = memref.alloc() : memref<64xf32>
  %B = memref.alloc() : memref<64xf32>
  affine.for %i = 0 to 64 {
    affine.store %z, %A[%i] : memref<64xf32>
  }
  %peek = memref.load %A[%c0] : memref<64xf32>
  affine.for %i = 0 to 64 {
    %a = affine.load %A[%i] : memref<64xf32>
    %s = arith.addf %a, %peek : f32
    affine.store %s, %B[%i] : memref<64xf32>
  }
  return
}
