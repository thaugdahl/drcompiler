// RUN: dr-opt %s --pass-pipeline='builtin.module(fold-memref-alias-ops,func.func(dr-raise-scf-to-affine,affine-raise-from-memref),canonicalize)' | FileCheck %s

// The whole bridge, on the exact shape a MARCO equation lowers to.  This pass
// only does the loops; the other two rewrites are upstream and can only run
// once the loops are affine (an scf.for induction variable is not a valid
// affine dimension):
//
//   fold-memref-alias-ops     rank-0 memref.subview + store  -> indexed store
//   affine-raise-from-memref  arith.addi %iv, %c             -> folded into the map
//
// Verified end to end on a real MARCO dump (ThermalChipOO 4^3, -O2):
// 173 scf.for / 3 affine.for before, 0 scf.for / 176 affine.for after.

// CHECK-LABEL: func.func @eq
// CHECK: affine.for %[[I:.*]] = 2 to 5 {
// CHECK:   affine.for %[[J:.*]] = 1 to 5 {
// CHECK:     affine.for %[[K:.*]] = 1 to 5 {
// CHECK:       %[[A:.*]] = affine.load %arg0[%[[I]] - 2, %[[J]] - 1, %[[K]] - 1]
// CHECK:       %[[B:.*]] = affine.load %arg0[%[[I]], %[[J]], %[[K]]]
// CHECK:       %[[D:.*]] = arith.subf %[[A]], %[[B]]
// CHECK:       affine.store %[[D]], %arg1[%[[I]] - 2, %[[J]] - 1, %[[K]] - 1]
// CHECK-NOT: scf.for
// CHECK-NOT: memref.subview
// CHECK-NOT: memref.load
func.func @eq(%T: memref<8x8x8xf64>, %Q: memref<8x8x8xf64>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index
  %cm1 = arith.constant -1 : index
  %cm2 = arith.constant -2 : index
  scf.for %i = %c2 to %c5 step %c1 {
    scf.for %j = %c1 to %c5 step %c1 {
      scf.for %k = %c1 to %c5 step %c1 {
        %a = arith.addi %i, %cm2 : index
        %b = arith.addi %j, %cm1 : index
        %c = arith.addi %k, %cm1 : index
        %t0 = memref.load %T[%a, %b, %c] : memref<8x8x8xf64>
        %t1 = memref.load %T[%i, %j, %k] : memref<8x8x8xf64>
        %d = arith.subf %t0, %t1 : f64
        %sv = memref.subview %Q[%a, %b, %c] [1, 1, 1] [1, 1, 1]
            : memref<8x8x8xf64> to memref<f64, strided<[], offset: ?>>
        memref.store %d, %sv[] : memref<f64, strided<[], offset: ?>>
      }
    }
  }
  return
}
