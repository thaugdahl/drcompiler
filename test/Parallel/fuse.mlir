// RUN: dr-opt %s --pass-pipeline='builtin.module(dr-par-bubbles{par-materialize})' | FileCheck %s

// M3: conformant depth-1 parallel sibling bands are fused into one par.forall
// when their cross-dependences are disjoint or element-aligned; an offset
// producer->consumer dependence becomes a par.barrier between two par.forall.

// Disjoint buffers (C vs T,A) -> ONE par.forall, bodies sequenced.
// CHECK-LABEL: func.func @fuse_disjoint
// CHECK:         par.region {
// CHECK:           par.forall([0], [128], [1]) {
// CHECK:             memref.store
// CHECK:             memref.load
// CHECK:             math.exp
// CHECK:             memref.store
// CHECK:             par.yield
// CHECK:           }
// CHECK-NOT:       par.forall
func.func @fuse_disjoint() {
  %z = arith.constant 0.0 : f32
  %C = memref.alloc() : memref<128xf32>
  %T = memref.alloc() : memref<128xf32>
  %A = memref.alloc() : memref<128xf32>
  affine.for %i = 0 to 128 { affine.store %z, %C[%i] : memref<128xf32> }
  affine.for %i = 0 to 128 {
    %a = affine.load %A[%i] : memref<128xf32>
    %e = math.exp %a : f32
    affine.store %e, %T[%i] : memref<128xf32>
  }
  return
}

// Same buffer, element-aligned (C[i] vs C[i]) -> ONE par.forall.
// CHECK-LABEL: func.func @fuse_aligned
// CHECK:         par.forall([0], [128], [1]) {
// CHECK:           memref.store %{{.*}}, %[[C:.*]][%[[I:.*]]]
// CHECK:           memref.load %[[C]][%[[I]]]
// CHECK:           arith.addf
// CHECK:           memref.store %{{.*}}, %[[C]][%[[I]]]
// CHECK-NOT:       par.forall
func.func @fuse_aligned() {
  %z = arith.constant 0.0 : f32
  %c1 = arith.constant 1.0 : f32
  %C = memref.alloc() : memref<128xf32>
  affine.for %i = 0 to 128 { affine.store %z, %C[%i] : memref<128xf32> }
  affine.for %i = 0 to 128 {
    %c = affine.load %C[%i] : memref<128xf32>
    %s = arith.addf %c, %c1 : f32
    affine.store %s, %C[%i] : memref<128xf32>
  }
  return
}

// Same buffer, offset (T[i] vs T[i+1]) -> par.forall, par.barrier, par.forall.
// CHECK-LABEL: func.func @offset
// CHECK:         par.region {
// CHECK:           par.forall([0], [100], [1]) {
// CHECK:             memref.store
// CHECK:             par.yield
// CHECK:           }
// CHECK:           par.barrier
// CHECK:           par.forall([0], [100], [1]) {
// CHECK:             memref.load
// CHECK:             memref.store
// CHECK:             par.yield
func.func @offset() {
  %z = arith.constant 0.0 : f32
  %T = memref.alloc() : memref<128xf32>
  %O = memref.alloc() : memref<128xf32>
  affine.for %i = 0 to 100 { affine.store %z, %T[%i] : memref<128xf32> }
  affine.for %i = 0 to 100 {
    %t = affine.load %T[%i + 1] : memref<128xf32>
    affine.store %t, %O[%i] : memref<128xf32>
  }
  return
}
