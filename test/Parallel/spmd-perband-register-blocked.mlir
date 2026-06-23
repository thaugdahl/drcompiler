// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block{cache-tile=true}),dr-par-bubbles{par-spmd-perband})' | FileCheck %s
// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block{cache-tile=true}),dr-par-bubbles{par-spmd-perband},func.func(convert-par-to-scf))' | FileCheck %s --check-prefix=SCF

// SPMD x vectorization compose: affine-register-block rewrites the GEMM into a
// stepped, unroll-and-jammed band with affine.vector_load/store + vector iter_arg
// accumulators (the per-thread vectorized kernel).  par-spmd-perband must shard
// that REGISTER-BLOCKED band into a par.forall (de-affining affine.vector_load/
// store -> vector.load/store), not fall back to par.critical -- so each SPMD
// shard runs the vectorized kernel (codegen x parallelism).  See
// PARALLEL_SPMD_SPEC.md §11.16.

// CHECK-LABEL: func.func @gemm
// CHECK:         par.region {
// CHECK:           par.forall(
// CHECK-NOT:       par.critical
// CHECK:             vector.load
// CHECK:             vector.store
// CHECK:         }

// SCF-LABEL: func.func @gemm
// SCF:         scf.parallel
// SCF:           vector.load
// SCF-NOT:       affine.vector_load
func.func @gemm(%A: memref<256x256xf32>, %B: memref<256x256xf32>, %C: memref<256x256xf32>) {
  affine.for %i = 0 to 256 {
    affine.for %j = 0 to 256 {
      affine.for %k = 0 to 256 {
        %a = affine.load %A[%i, %k] : memref<256x256xf32>
        %b = affine.load %B[%k, %j] : memref<256x256xf32>
        %c = affine.load %C[%i, %j] : memref<256x256xf32>
        %p = arith.mulf %a, %b : f32
        %s = arith.addf %c, %p : f32
        affine.store %s, %C[%i, %j] : memref<256x256xf32>
      }
    }
  }
  return
}
