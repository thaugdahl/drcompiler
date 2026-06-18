// RUN: dr-opt %s --pass-pipeline='builtin.module(dr-par-bubbles{par-materialize}, func.func(convert-par-to-scf))' | FileCheck %s

// M2 sink: materialize, then lower the `par` dialect to scf (sequential
// reference).  par.forall -> scf.parallel, par.region inlined, par.yield erased.

// CHECK-LABEL: func.func @gemm
// CHECK:         scf.parallel (%[[I:.*]], %[[J:.*]]) =
// CHECK:           scf.for %[[K:.*]] =
// CHECK:             memref.load
// CHECK:             memref.store
// CHECK:           scf.reduce
// CHECK-NOT:     par.region
// CHECK-NOT:     par.forall
// CHECK-NOT:     par.yield
func.func @gemm() {
  %A = memref.alloc() : memref<64x64xf32>
  %B = memref.alloc() : memref<64x64xf32>
  %C = memref.alloc() : memref<64x64xf32>
  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 64 {
      affine.for %k = 0 to 64 {
        %a = affine.load %A[%i, %k] : memref<64x64xf32>
        %b = affine.load %B[%k, %j] : memref<64x64xf32>
        %c = affine.load %C[%i, %j] : memref<64x64xf32>
        %p = arith.mulf %a, %b : f32
        %s = arith.addf %c, %p : f32
        affine.store %s, %C[%i, %j] : memref<64x64xf32>
      }
    }
  }
  return
}
