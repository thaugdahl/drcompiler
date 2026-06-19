// RUN: dr-opt %s -split-input-file --pass-pipeline='builtin.module(dr-par-bubbles{par-spmd})' | FileCheck %s
// RUN: dr-opt %s -split-input-file --pass-pipeline='builtin.module(dr-par-bubbles{par-spmd},func.func(convert-par-to-scf))' | FileCheck %s --check-prefix=SCF

// S2 (PARALLEL_SPMD_SPEC.md §7): widen every top-level shard-axis band into ONE
// par.region, hoist one par.forall over the shard axis (ELIDE-connected runs
// share a forall, owner-computes; inner loops -> scf.for), par.barrier /
// par.redistribute only at non-elided edges.  par->scf gives the sequential
// reference.

// Owner-aligned eltwise chain over one axis -> one forall, zero barriers
// (every edge ELIDEs, so all three bands fuse into one shard loop).
// CHECK-LABEL: func.func @spmd_elide
// CHECK:         par.region {
// CHECK:           par.forall([0], [64], [1]) {
// CHECK-NOT:       par.barrier
// CHECK-NOT:       par.forall
// CHECK:         }
// SCF-LABEL: func.func @spmd_elide
// SCF:         scf.parallel
// SCF-NOT:     scf.parallel
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
  affine.for %i = 0 to 64 {
    %b = affine.load %B[%i] : memref<64xf32>
    %t = math.tanh %b : f32
    affine.store %t, %C[%i] : memref<64xf32>
  }
  affine.for %i = 0 to 64 {
    %c = affine.load %C[%i] : memref<64xf32>
    %s = arith.addf %c, %z : f32
    affine.store %s, %A[%i] : memref<64xf32>
  }
  return
}

// -----

// Multi-dim shard (batch axis n, inner spatial j): two owner-aligned bands ELIDE
// into one forall (owner-computes: worker n does exp then tanh on its row); a
// third band transpose-reads C (D[n,j] = C[j,n], so worker n needs every other
// worker's row) -- a genuine cross-shard dependence -> a second forall behind a
// par.barrier.  This is the unique S2 case (non-fusable multi-dim) M3 cannot
// express: M3 only fuses conformant depth-1 siblings, never widens a multi-dim
// band into a whole-function owner-computes region.
// CHECK-LABEL: func.func @spmd_multidim
// CHECK:         par.region {
// CHECK:           par.forall([0], [8], [1]) {
// CHECK:           ^bb0(%[[N1:.*]]: index):
// CHECK:             scf.for %[[J1:.*]] =
// CHECK:               memref.load %{{.*}}[%[[N1]], %[[J1]]]
// CHECK:             scf.for %[[J2:.*]] =
// CHECK:             par.yield
// CHECK:           }
// CHECK:           par.barrier
// CHECK:           par.forall([0], [8], [1]) {
// CHECK:             par.yield
// CHECK:           }
// CHECK:           par.yield
// CHECK:         }
// SCF-LABEL: func.func @spmd_multidim
// SCF:         scf.parallel
// SCF:         scf.parallel
func.func @spmd_multidim() {
  %A = memref.alloc() : memref<8x8xf32>
  %B = memref.alloc() : memref<8x8xf32>
  %C = memref.alloc() : memref<8x8xf32>
  %D = memref.alloc() : memref<8x8xf32>
  affine.for %n = 0 to 8 {
    affine.for %j = 0 to 8 {
      %a = affine.load %A[%n, %j] : memref<8x8xf32>
      %e = math.exp %a : f32
      affine.store %e, %B[%n, %j] : memref<8x8xf32>
    }
  }
  affine.for %n = 0 to 8 {
    affine.for %j = 0 to 8 {
      %b = affine.load %B[%n, %j] : memref<8x8xf32>
      %t = math.tanh %b : f32
      affine.store %t, %C[%n, %j] : memref<8x8xf32>
    }
  }
  affine.for %n = 0 to 8 {
    affine.for %j = 0 to 8 {
      %c = affine.load %C[%j, %n] : memref<8x8xf32>
      affine.store %c, %D[%n, %j] : memref<8x8xf32>
    }
  }
  return
}

// -----

// Real-ONNX shape: each layer's scratch buffer (and a per-layer constant) is
// allocated BETWEEN the bands.  The glue is hoisted above the first band so the
// run is contiguous, then the owner-aligned layers fuse into ONE forall.  All
// three bands ELIDE -> zero barriers.
// CHECK-LABEL: func.func @interleaved_glue
// CHECK:         memref.alloc
// CHECK:         memref.alloc
// CHECK:         par.region {
// CHECK:           par.forall([0], [8], [1]) {
// CHECK-NOT:       par.barrier
// CHECK-NOT:       par.forall
// CHECK:         }
func.func @interleaved_glue(%A: memref<8x8xf32>, %D: memref<8x8xf32>) {
  %B = memref.alloc() : memref<8x8xf32>
  affine.for %n = 0 to 8 { affine.for %j = 0 to 8 {
    %a = affine.load %A[%n, %j] : memref<8x8xf32>
    affine.store %a, %B[%n, %j] : memref<8x8xf32>
  }}
  %C = memref.alloc() : memref<8x8xf32>
  %k = arith.constant 2.0 : f32
  affine.for %n = 0 to 8 { affine.for %j = 0 to 8 {
    %b = affine.load %B[%n, %j] : memref<8x8xf32>
    %m = arith.mulf %b, %k : f32
    affine.store %m, %C[%n, %j] : memref<8x8xf32>
  }}
  affine.for %n = 0 to 8 { affine.for %j = 0 to 8 {
    %c = affine.load %C[%n, %j] : memref<8x8xf32>
    affine.store %c, %D[%n, %j] : memref<8x8xf32>
  }}
  return
}

// -----

// Dynamic (runtime) batch extent `0 to %N`, shared across both owner-aligned
// bands -> ONE par.forall over the kDynamic shard axis carrying %N as the
// dynamic upper bound.  This is the real-ONNX batch-throughput shape.
// CHECK-LABEL: func.func @dyn_batch
// CHECK:         par.region {
// CHECK:           par.forall([0], [-9223372036854775808], [1]) dyn(%arg2 : index) {
// CHECK-NOT:       par.barrier
// CHECK-NOT:       par.forall
// CHECK:         }
func.func @dyn_batch(%A: memref<?x16xf32>, %D: memref<?x16xf32>, %N: index) {
  %B = memref.alloc(%N) : memref<?x16xf32>
  affine.for %n = 0 to %N { affine.for %j = 0 to 16 {
    %a = affine.load %A[%n, %j] : memref<?x16xf32>
    %e = arith.mulf %a, %a : f32
    affine.store %e, %B[%n, %j] : memref<?x16xf32>
  }}
  affine.for %n = 0 to %N { affine.for %j = 0 to 16 {
    %b = affine.load %B[%n, %j] : memref<?x16xf32>
    affine.store %b, %D[%n, %j] : memref<?x16xf32>
  }}
  return
}
