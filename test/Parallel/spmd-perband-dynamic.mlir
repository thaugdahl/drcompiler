// RUN: dr-opt %s --pass-pipeline='builtin.module(dr-par-bubbles{par-spmd-perband})' | FileCheck %s
// RUN: dr-opt %s --pass-pipeline='builtin.module(dr-par-bubbles{par-spmd-perband})' | FileCheck %s --check-prefix=BAL
// RUN: dr-opt %s --pass-pipeline='builtin.module(dr-par-bubbles{par-spmd-perband},func.func(convert-par-to-omp))' | FileCheck %s --check-prefix=OMP

// Load-imbalanced (triangular) band: the parallel shard axis `i` owns rows whose
// inner extent `0..i+1` GROWS with i, so a static block schedule starves the
// low-i workers.  The forall is tagged `par.dynamic` and lowers to an
// omp.wsloop with a dynamic schedule (workers pull rows on demand).  A balanced
// (constant-extent) band is NOT tagged -> default static schedule.
// CHECK-LABEL: func.func @tri
// CHECK:         par.forall([0], [64], [1]) {
// CHECK:         } {par.dynamic}
// OMP-LABEL: func.func @tri
// OMP:         omp.wsloop schedule(dynamic)
#map = affine_map<(d0) -> (d0 + 1)>
func.func @tri(%C: memref<64x64xf64>, %A: memref<64x64xf64>) {
  affine.for %i = 0 to 64 {
    affine.for %j = 0 to #map(%i) {
      %a = affine.load %A[%i, %j] : memref<64x64xf64>
      %c = affine.load %C[%i, %j] : memref<64x64xf64>
      %s = arith.addf %c, %a : f64
      affine.store %s, %C[%i, %j] : memref<64x64xf64>
    }
  }
  return
}

// CHECK-LABEL: func.func @bal
// BAL-LABEL: func.func @bal
// BAL:         par.forall([0], [64], [1]) {
// BAL-NOT:     par.dynamic
func.func @bal(%C: memref<64x64xf64>, %A: memref<64x64xf64>) {
  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 64 {
      %a = affine.load %A[%i, %j] : memref<64x64xf64>
      %c = affine.load %C[%i, %j] : memref<64x64xf64>
      %s = arith.addf %c, %a : f64
      affine.store %s, %C[%i, %j] : memref<64x64xf64>
    }
  }
  return
}
