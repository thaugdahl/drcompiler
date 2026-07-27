// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-raise-scf-to-affine))' | FileCheck %s

// The shape MARCO emits for a 3-D equation once the equation call has been
// inlined: rectangular, unit step, constant bounds, no iter_args.  All three
// loops raise; the body moves across untouched (the index arithmetic and the
// memref accesses are upstream's job, see the bridge test).

// CHECK-LABEL: func.func @constant_bounds
// CHECK: affine.for %[[I:.*]] = 2 to 5 {
// CHECK:   affine.for %[[J:.*]] = 1 to 5 {
// CHECK:     affine.for %[[K:.*]] = 1 to 5 {
// CHECK:       memref.load
// CHECK-NOT: scf.for
func.func @constant_bounds(%T: memref<8x8x8xf64>, %Q: memref<8x8x8xf64>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index
  scf.for %i = %c2 to %c5 step %c1 {
    scf.for %j = %c1 to %c5 step %c1 {
      scf.for %k = %c1 to %c5 step %c1 {
        %v = memref.load %T[%i, %j, %k] : memref<8x8x8xf64>
        memref.store %v, %Q[%i, %j, %k] : memref<8x8x8xf64>
      }
    }
  }
  return
}

// A non-unit but constant step is fine — affine.for carries it verbatim.

// CHECK-LABEL: func.func @constant_step
// CHECK: affine.for %{{.*}} = 0 to 64 step 4 {
// CHECK-NOT: scf.for
func.func @constant_step(%A: memref<64xf64>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.0 : f64
  scf.for %i = %c0 to %c64 step %c4 {
    memref.store %cst, %A[%i] : memref<64xf64>
  }
  return
}
