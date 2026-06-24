// RUN: dr-opt %s --pass-pipeline='builtin.module(dr-par-bubbles{par-spmd-perband})' | FileCheck %s
// RUN: dr-opt %s --pass-pipeline='builtin.module(dr-par-bubbles{par-spmd-perband})' -verify-diagnostics
// RUN: dr-opt %s --pass-pipeline='builtin.module(dr-par-bubbles{par-spmd-perband},func.func(convert-par-to-scf))' | FileCheck %s --check-prefix=SCF
// RUN: dr-opt %s --pass-pipeline='builtin.module(dr-par-bubbles{par-spmd-perband},func.func(convert-par-to-omp))' | FileCheck %s --check-prefix=OMP
// par-spmd-diag characterizes each CRITICAL band (loop bounds + AxisKind + op
// histogram) -- the serial-floor scout.  @with_critical's band carries B[i]<-B[i-1].
// RUN: dr-opt %s --pass-pipeline='builtin.module(dr-par-bubbles{par-spmd-perband par-spmd-diag})' 2>&1 | FileCheck %s --check-prefix=DIAG
// DIAG: par-spmd-critical: loops[1:64(Cd)] ops{affine.load:1 arith.addf:1 affine.store:1}
// par-spmd-axes scouts shard-axis CONSISTENCY: each band's parallel-axis extents
// + the extent perBandShard chose (greedy outermost).  @perlayer band 2 has both
// 32 and 64 parallel but greedily shards 32.  (A global-dominant-axis preference
// was measured WORSE on openai-gpt -- §11.21 -- so the greedy choice stands.)
// RUN: dr-opt %s --pass-pipeline='builtin.module(dr-par-bubbles{par-spmd-perband par-spmd-diag})' 2>&1 | FileCheck %s --check-prefix=AXES
// AXES-DAG: par-spmd-axes: par=[64,64] chosen=64
// AXES-DAG: par-spmd-axes: par=[32,64] chosen=32

// S7 batch-1 within-sample: each band shards its OWN outermost parallel loop
// (extents differ per layer: 64, 32, 32), one par.region.  Layer 2 reads B
// across the shard axis (B[63-i]) -- a true cross-band dependence that KEEPS a
// par.barrier (and the extent changes 64->32, a different owner partition).
// Layer 3 reads C[i] exactly as layer 2 wrote it (same map, same 32-extent
// partition) -> owner-aligned -> the barrier ELIDES (PARALLEL_SPMD_SPEC.md §4).
// par->scf gives the sequential reference.
// CHECK-LABEL: func.func @perlayer
// CHECK:         par.region {
// CHECK:           par.forall([0], [64], [1]) {
// CHECK:             scf.for
// CHECK:           par.barrier
// CHECK:           par.forall([0], [32], [1]) {
// CHECK-NOT:         par.barrier
// CHECK:           par.forall([0], [32], [1]) {
// CHECK:           par.yield
// CHECK:         }
// SCF-LABEL: func.func @perlayer
// SCF:         scf.parallel
// SCF:         scf.parallel
// SCF:         scf.parallel
// The kept barrier lowers to omp.barrier; the elided edge -> nowait wsloop, no
// barrier between the two 32-extent loops.
// OMP-LABEL: func.func @perlayer
// OMP:         omp.parallel {
// OMP:           omp.wsloop nowait {
// OMP:           omp.barrier
// OMP:           omp.wsloop nowait {
// OMP-NOT:       omp.barrier
// OMP:           omp.terminator
// expected-remark@+1 {{par-spmd-perband: materialized foralls=3 critical=0 replicated=0 moved=0 barriers=1 elided=1 (3 bands, 3 parallel)}}
func.func @perlayer(%A: memref<64x64xf32>, %D: memref<32x64xf32>) {
  %B = memref.alloc() : memref<64x64xf32>
  %C = memref.alloc() : memref<32x64xf32>
  %one = arith.constant 1.0 : f32
  %three = arith.constant 3.0 : f32
  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 64 {
      %a = affine.load %A[%i, %j] : memref<64x64xf32>
      %b = arith.addf %a, %one : f32
      affine.store %b, %B[%i, %j] : memref<64x64xf32>
    }
  }
  affine.for %i = 0 to 32 {
    affine.for %j = 0 to 64 {
      %b0 = affine.load %B[%i, %j] : memref<64x64xf32>
      %b1 = affine.load %B[63 - %i, %j] : memref<64x64xf32>
      %s = arith.addf %b0, %b1 : f32
      affine.store %s, %C[%i, %j] : memref<32x64xf32>
    }
  }
  affine.for %i = 0 to 32 {
    affine.for %j = 0 to 64 {
      %c = affine.load %C[%i, %j] : memref<32x64xf32>
      %d = arith.mulf %c, %three : f32
      affine.store %d, %D[%i, %j] : memref<32x64xf32>
    }
  }
  return
}

// -----

// A non-shardable band (outer loop carries a dependence: B[i] <- B[i-1]) falls
// back to par.critical (single worker) so the function still materializes into
// one team; the parallel band before it is a par.forall.  The forall->critical
// edge is NOT elided (a critical reads single-worker; a barrier guards it).
// CHECK-LABEL: func.func @with_critical
// CHECK:         par.region {
// CHECK:           par.forall([0], [64], [1]) {
// CHECK:           par.barrier
// CHECK:           par.critical {
// CHECK:             affine.for %{{.*}} = 1 to 64 {
// CHECK:             par.yield
// CHECK:           par.yield
// CHECK:         }
// expected-remark@+1 {{par-spmd-perband: materialized foralls=1 critical=1 replicated=0 moved=0 barriers=1 elided=0 (2 bands, 1 parallel)}}
func.func @with_critical(%A: memref<64xf32>, %B: memref<64xf32>) {
  %one = arith.constant 1.0 : f32
  affine.for %i = 0 to 64 {
    %a = affine.load %A[%i] : memref<64xf32>
    %b = arith.addf %a, %one : f32
    affine.store %b, %B[%i] : memref<64xf32>
  }
  affine.for %i = 1 to 64 {
    %p = affine.load %B[%i - 1] : memref<64xf32>
    %v = arith.addf %p, %one : f32
    affine.store %v, %B[%i] : memref<64xf32>
  }
  return
}

// -----

// Inter-band glue (a scalar read of band-1 output + a multiply) is REPLICATED
// at region level between the foralls -- every worker recomputes it, so the
// SSA value %f is visible inside the second forall (whole-function widening).
// The forall->forall edge (band 2 reads B[i] as band 1 wrote it, same 64-extent
// partition) is owner-aligned -> the par.barrier ELIDES.  Soundness of the
// glue's cross-shard read of B[0]: it sits between the foralls, so band 1's
// wsloop is NOT nowait (its next op is the glue, not a forall/barrier) -> its
// implicit end barrier syncs band 1 before any worker reads B[0].
// CHECK-LABEL: func.func @glue
// CHECK:         par.region {
// CHECK:           par.forall([0], [64], [1]) {
// CHECK:             par.yield
// CHECK:           }
// CHECK:           memref.load
// CHECK:           arith.mulf
// CHECK-NOT:       par.barrier
// CHECK:           par.forall([0], [64], [1]) {
// The glue read of B[0] is guarded by band 1's implicit barrier (not nowait).
// OMP-LABEL: func.func @glue
// OMP:         omp.parallel {
// OMP:           omp.wsloop {
// OMP-NOT:         nowait
// OMP:           memref.load
// OMP:           omp.wsloop
// expected-remark@+1 {{par-spmd-perband: materialized foralls=2 critical=0 replicated=2 moved=0 barriers=0 elided=1 (2 bands, 2 parallel)}}
func.func @glue(%A: memref<64xf32>, %D: memref<64xf32>) {
  %c0 = arith.constant 0 : index
  %two = arith.constant 2.0 : f32
  %one = arith.constant 1.0 : f32
  %B = memref.alloc() : memref<64xf32>
  affine.for %i = 0 to 64 {
    %a = affine.load %A[%i] : memref<64xf32>
    %b = arith.addf %a, %one : f32
    affine.store %b, %B[%i] : memref<64xf32>
  }
  %s = memref.load %B[%c0] : memref<64xf32>
  %f = arith.mulf %s, %two : f32
  affine.for %i = 0 to 64 {
    %b = affine.load %B[%i] : memref<64xf32>
    %d = arith.mulf %b, %f : f32
    affine.store %d, %D[%i] : memref<64xf32>
  }
  return
}
