// RUN: dr-opt %s --pass-pipeline='builtin.module(dr-par-bubbles{par-spmd-perband})' | FileCheck %s
// RUN: dr-opt %s --pass-pipeline='builtin.module(dr-par-bubbles{par-spmd-perband},func.func(convert-par-to-omp))' | FileCheck %s --check-prefix=OMP
// RUN: dr-opt %s --pass-pipeline='builtin.module(dr-par-bubbles{par-spmd-perband},func.func(convert-par-to-scf))' | FileCheck %s --check-prefix=SCF

// SEQWRAP: a time-stepped stencil -- the outer `t` loop is SEQUENTIAL (A<->B is
// carried across t) but wraps two PARALLEL spatial bands.  Materialized as ONE
// team (par.region) holding scf.for(t) { par.forall ; par.forall } -- the team
// persists across timesteps and re-distributes the spatial work each step.  No
// par.barrier (it may not nest under scf.for); consecutive bands synchronize via
// the implicit end-of-wsloop barrier (each omp.wsloop is emitted WITHOUT nowait).
// CHECK-LABEL: func.func @stencil
// CHECK:         par.region {
// CHECK:           scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:             par.forall([1], [63], [1]) {
// CHECK:               scf.for
// CHECK:             par.forall([1], [63], [1]) {
// CHECK:               scf.for
// CHECK:           par.yield

// one omp.parallel; the sequential scf.for(t) holds two omp.wsloops, and
// NEITHER carries `nowait` (the implicit end barrier syncs the two bands).
// OMP-LABEL: func.func @stencil
// OMP:         omp.parallel {
// OMP:           scf.for
// OMP:             omp.wsloop {
// OMP:             omp.wsloop {
// OMP-NOT:       omp.wsloop nowait

// SCF reference: sequential time loop with two scf.parallel spatial sweeps.
// SCF-LABEL: func.func @stencil
// SCF:         scf.for
// SCF:           scf.parallel
// SCF:           scf.parallel
func.func @stencil(%A: memref<64x64xf64>, %B: memref<64x64xf64>) {
  %c = arith.constant 0.2 : f64
  affine.for %t = 0 to 100 {
    affine.for %i = 1 to 63 {
      affine.for %j = 1 to 63 {
        %0 = affine.load %A[%i, %j] : memref<64x64xf64>
        %1 = affine.load %A[%i - 1, %j] : memref<64x64xf64>
        %2 = affine.load %A[%i + 1, %j] : memref<64x64xf64>
        %3 = arith.addf %0, %1 : f64
        %4 = arith.addf %3, %2 : f64
        %5 = arith.mulf %4, %c : f64
        affine.store %5, %B[%i, %j] : memref<64x64xf64>
      }
    }
    affine.for %i = 1 to 63 {
      affine.for %j = 1 to 63 {
        %0 = affine.load %B[%i, %j] : memref<64x64xf64>
        %1 = affine.load %B[%i, %j - 1] : memref<64x64xf64>
        %2 = affine.load %B[%i, %j + 1] : memref<64x64xf64>
        %3 = arith.addf %0, %1 : f64
        %4 = arith.addf %3, %2 : f64
        %5 = arith.mulf %4, %c : f64
        affine.store %5, %A[%i, %j] : memref<64x64xf64>
      }
    }
  }
  return
}
