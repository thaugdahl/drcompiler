// RUN: dr-opt %s --pass-pipeline='builtin.module(dr-par-bubbles{par-spmd-perband})' | FileCheck %s
// RUN: dr-opt %s --pass-pipeline='builtin.module(dr-par-bubbles{par-spmd-perband},func.func(convert-par-to-scf))' | FileCheck %s --check-prefix=SCF

// Imperfect parallel band: the outer `i` loop is parallel (each row of C is
// owned), but its body is NOT a perfect nest -- it holds two sibling sub-nests
// (a beta-scale of row i, then a k/j accumulation into row i).  The shard loop
// is de-affined into the forall (affine.for -> scf.for, affine.load/store ->
// memref) so the band materializes as ONE par.forall instead of par.critical.
// This is the function-argument-memref kernel shape (PolyBench gemm).
// CHECK-LABEL: func.func @gemm_like
// CHECK:         par.region {
// CHECK:           par.forall([0], [64], [1]) {
// CHECK:           ^bb0(%[[IV:.*]]: index):
// CHECK:             scf.for
// CHECK:               memref.load
// CHECK:               memref.store
// CHECK:             scf.for
// CHECK:               scf.for
// CHECK:                 memref.load
// CHECK:                 memref.store
// CHECK:             par.yield
// CHECK:           par.yield
// CHECK:         }
// SCF-LABEL: func.func @gemm_like
// SCF:         scf.parallel
// SCF-NOT:     affine.for
func.func @gemm_like(%C: memref<64x64xf64>, %A: memref<64x64xf64>,
                     %B: memref<64x64xf64>, %beta: f64) {
  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 64 {
      %c = affine.load %C[%i, %j] : memref<64x64xf64>
      %s = arith.mulf %c, %beta : f64
      affine.store %s, %C[%i, %j] : memref<64x64xf64>
    }
    affine.for %k = 0 to 64 {
      affine.for %j = 0 to 64 {
        %a = affine.load %A[%i, %k] : memref<64x64xf64>
        %b = affine.load %B[%k, %j] : memref<64x64xf64>
        %p = arith.mulf %a, %b : f64
        %c = affine.load %C[%i, %j] : memref<64x64xf64>
        %s = arith.addf %c, %p : f64
        affine.store %s, %C[%i, %j] : memref<64x64xf64>
      }
    }
  }
  return
}
