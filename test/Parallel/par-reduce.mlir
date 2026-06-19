// RUN: dr-opt %s | FileCheck %s
// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(convert-par-to-scf))' | FileCheck %s --check-prefix=SCF

// par.reduce (PARALLEL_PAR_DIALECT_SPEC.md §3.6): a reducing par.forall carries
// init/result values; the terminator contributes one value per result through a
// kind-named associative combinator.  par->scf is the SEQUENTIAL reference: an
// scf.for with iter_args, exact (no FP reassociation).

// Sum reduction (kind 0 = addf).
// CHECK-LABEL: func.func @sum
// CHECK:         %[[R:.*]] = par.forall([0], [64], [1]) init(%{{.*}} : f32) {
// CHECK:           par.reduce(%{{.*}} : f32) kinds [0]
// CHECK:         } -> f32
// SCF-LABEL: func.func @sum
// SCF:         %[[S:.*]] = scf.for %{{.*}} iter_args(%[[A:.*]] = %{{.*}}) -> (f32)
// SCF:           arith.addf %[[A]], %{{.*}} : f32
// SCF:           scf.yield
func.func @sum(%A: memref<64xf32>, %out: memref<f32>) {
  %z = arith.constant 0.0 : f32
  %s = par.forall([0], [64], [1]) init(%z : f32) {
  ^bb0(%i: index):
    %a = memref.load %A[%i] : memref<64xf32>
    par.reduce(%a : f32) kinds [0]
  } -> f32
  memref.store %s, %out[] : memref<f32>
  return
}

// -----

// Max reduction (kind 2 = maxnumf) over a dynamic extent.
// SCF-LABEL: func.func @maxred
// SCF:         scf.for %{{.*}} to %{{.*}} step
// SCF:           arith.maxnumf
func.func @maxred(%A: memref<?xf32>, %N: index, %out: memref<f32>) {
  %ninf = arith.constant -3.40282347E+38 : f32
  %m = par.forall([0], [-9223372036854775808], [1]) dyn(%N : index) init(%ninf : f32) {
  ^bb0(%i: index):
    %a = memref.load %A[%i] : memref<?xf32>
    par.reduce(%a : f32) kinds [2]
  } -> f32
  memref.store %m, %out[] : memref<f32>
  return
}
