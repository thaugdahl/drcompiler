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
