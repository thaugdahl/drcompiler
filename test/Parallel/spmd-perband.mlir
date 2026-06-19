// RUN: dr-opt %s --pass-pipeline='builtin.module(dr-par-bubbles{par-spmd-perband})' | FileCheck %s
// RUN: dr-opt %s --pass-pipeline='builtin.module(dr-par-bubbles{par-spmd-perband},func.func(convert-par-to-scf))' | FileCheck %s --check-prefix=SCF

// S7 batch-1 within-sample: each band shards its OWN outermost parallel loop
// (extents differ per layer: 64, 32, 32), one par.region, par.barrier between
// bands.  Layer 2 reads B across the shard axis (B[63-i]) -- a true cross-band
// dependence the barrier covers.  par->scf gives the sequential reference.
// CHECK-LABEL: func.func @perlayer
// CHECK:         par.region {
// CHECK:           par.forall([0], [64], [1]) {
// CHECK:             scf.for
// CHECK:           par.barrier
// CHECK:           par.forall([0], [32], [1]) {
// CHECK:           par.barrier
// CHECK:           par.forall([0], [32], [1]) {
// CHECK:           par.yield
// CHECK:         }
// SCF-LABEL: func.func @perlayer
// SCF:         scf.parallel
// SCF:         scf.parallel
// SCF:         scf.parallel
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
// one team; the parallel band before it is a par.forall.
// CHECK-LABEL: func.func @with_critical
// CHECK:         par.region {
// CHECK:           par.forall([0], [64], [1]) {
// CHECK:           par.barrier
// CHECK:           par.critical {
// CHECK:             affine.for %{{.*}} = 1 to 64 {
// CHECK:             par.yield
// CHECK:           par.yield
// CHECK:         }
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
