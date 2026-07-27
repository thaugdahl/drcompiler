// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-raise-scf-to-affine))' | FileCheck %s

// MARCO emits one func.func per equation, taking the iteration ranges as
// `index` block arguments and calling it from the schedule driver with
// constants.  Block arguments of a function are valid affine symbols, so the
// nest raises WITHOUT inlining first -- which matters because inlining 127
// equations is only needed for cross-equation fusion, not for tiling or
// distribution of a single equation.

// The identity symbol map prints in the shorthand `%arg0 to %arg1` form.

// CHECK-LABEL: func.func @equation
// CHECK-SAME: (%[[LB0:.*]]: index, %[[UB0:.*]]: index, %[[LB1:.*]]: index, %[[UB1:.*]]: index,
// CHECK: affine.for %{{.*}} = %[[LB0]] to %[[UB0]] {
// CHECK:   affine.for %{{.*}} = %[[LB1]] to %[[UB1]] {
// CHECK-NOT: scf.for
func.func @equation(%lb0: index, %ub0: index, %lb1: index, %ub1: index,
                    %A: memref<64x64xf64>) {
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.0 : f64
  scf.for %i = %lb0 to %ub0 step %c1 {
    scf.for %j = %lb1 to %ub1 step %c1 {
      memref.store %cst, %A[%i, %j] : memref<64x64xf64>
    }
  }
  return
}
