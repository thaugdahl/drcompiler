// RUN: dr-opt %s --pass-pipeline='builtin.module(dr-par-bubbles{par-materialize})' | FileCheck %s

// M2: materialize a maximal climb region into the `par` dialect.  The parallel
// prefix becomes a par.forall; the sequential suffix stays as scf.for; affine
// accesses are expanded to memref (par.forall IVs are not affine dims).

// CHECK-LABEL: func.func @init
// CHECK:         par.region {
// CHECK:           par.forall([0], [128], [1]) {
// CHECK:           ^bb0(%[[I:.*]]: index):
// CHECK:             memref.store %{{.*}}, %{{.*}}[%[[I]]] : memref<128xf32>
// CHECK:             par.yield
// CHECK:           }
// CHECK:           par.yield
// CHECK:         }
func.func @init() {
  %z = arith.constant 0.0 : f32
  %B = memref.alloc() : memref<128xf32>
  affine.for %i = 0 to 128 {
    affine.store %z, %B[%i] : memref<128xf32>
  }
  return
}

// GEMM (alloc'd buffers): i,j parallel -> par.forall(2-D); k carried -> scf.for.
// CHECK-LABEL: func.func @gemm
// CHECK:         par.region {
// CHECK:           par.forall([0, 0], [64, 64], [1, 1]) {
// CHECK:           ^bb0(%[[I:.*]]: index, %[[J:.*]]: index):
// CHECK:             scf.for %[[K:.*]] =
// CHECK:               memref.load %{{.*}}[%[[I]], %[[K]]]
// CHECK:               memref.load %{{.*}}[%[[K]], %[[J]]]
// CHECK:               memref.load %{{.*}}[%[[I]], %[[J]]]
// CHECK:               memref.store %{{.*}}, %{{.*}}[%[[I]], %[[J]]]
// CHECK:             }
// CHECK:             par.yield
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
