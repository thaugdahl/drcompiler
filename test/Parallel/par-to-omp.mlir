// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(convert-par-to-omp))' | FileCheck %s

// S3 (PARALLEL_SPMD_SPEC.md §7): the faithful par -> OpenMP lowering.
//   par.region      -> omp.parallel              (ONE team)
//   par.forall      -> omp.wsloop { omp.loop_nest }
//     elided edge    -> `nowait` (the explicit boundary omp.barrier syncs)
//   par.barrier      -> omp.barrier
//   par.redistribute -> omp.barrier (conservative)

// A forall before a barrier is `nowait` (no double barrier); the trailing forall
// keeps its implicit end barrier; one omp.parallel wraps the whole region.
// CHECK-LABEL: func.func @two_foralls
// CHECK:         omp.parallel {
// CHECK:           omp.wsloop nowait {
// CHECK:             omp.loop_nest (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) {
// CHECK:               memref.load
// CHECK:               omp.yield
// CHECK:           omp.barrier
// CHECK:           omp.wsloop {
// CHECK:             omp.loop_nest
// CHECK:           omp.terminator
// CHECK-NOT:     par.
func.func @two_foralls(%A: memref<8xf32>, %B: memref<8xf32>) {
  par.region {
    par.forall([0], [8], [1]) {
    ^bb0(%i: index):
      %a = memref.load %A[%i] : memref<8xf32>
      memref.store %a, %B[%i] : memref<8xf32>
      par.yield
    }
    par.barrier
    par.forall([0], [8], [1]) {
    ^bb0(%i: index):
      %b = memref.load %B[%i] : memref<8xf32>
      memref.store %b, %A[%i] : memref<8xf32>
      par.yield
    }
    par.yield
  }
  return
}

// -----

// A par.redistribute boundary also lowers to an omp.barrier (conservative).
// CHECK-LABEL: func.func @redistribute_barrier
// CHECK:         omp.parallel {
// CHECK:           omp.wsloop nowait {
// CHECK:           omp.barrier
// CHECK:           omp.wsloop {
// CHECK:           omp.terminator
func.func @redistribute_barrier(%A: memref<8xf32>, %B: memref<8xf32>) {
  par.region {
    par.forall([0], [8], [1]) {
    ^bb0(%i: index):
      %a = memref.load %A[%i] : memref<8xf32>
      memref.store %a, %B[%i] : memref<8xf32>
      par.yield
    }
    par.redistribute %B : memref<8xf32> from "row" to "col"
    par.forall([0], [8], [1]) {
    ^bb0(%i: index):
      %b = memref.load %B[%i] : memref<8xf32>
      memref.store %b, %A[%i] : memref<8xf32>
      par.yield
    }
    par.yield
  }
  return
}
