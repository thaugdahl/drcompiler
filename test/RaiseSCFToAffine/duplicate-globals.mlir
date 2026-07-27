// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-raise-scf-to-affine{emit-rationale=true}))' -verify-diagnostics -o /dev/null
// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-raise-scf-to-affine,affine-loop-invariant-code-motion,cse))' | FileCheck %s

// Raising the loops is necessary but NOT sufficient.
// AffineAnalysis.cpp checkMemrefAccessDependence:
//
//     if (srcAccess.memref != dstAccess.memref)
//       return DependenceResult::NoDependence;
//
// i.e. the dependence test compares memref SSA *values*.  MARCO re-fetches
// every array with its own memref.get_global inside every equation (609 of
// them for a 127-equation model over 16 distinct arrays), so two nests that
// read and write the same global through different get_global results are
// reported as INDEPENDENT.  That is a miscompile hazard for any transform that
// trusts the answer, not merely a missed fusion.
//
// The fix is upstream and must run AFTER raising, because LICM needs affine.for
// to hoist out of:  func.func(affine-loop-invariant-code-motion, cse)

memref.global "private" @T : memref<8xf64> = uninitialized
memref.global "private" @Q : memref<8xf64> = uninitialized

// expected-remark @below {{raised 2 scf.for, skipped 0}}
// expected-remark @below {{4 memref.get_global ops for 2 distinct globals are duplicated}}
func.func @two_equations() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %cst = arith.constant 2.000000e+00 : f64

  scf.for %i = %c0 to %c8 step %c1 {
    %t = memref.get_global @T : memref<8xf64>
    %v = memref.load %t[%i] : memref<8xf64>
    %w = arith.mulf %v, %cst : f64
    %q = memref.get_global @Q : memref<8xf64>
    memref.store %w, %q[%i] : memref<8xf64>
  }

  scf.for %i = %c0 to %c8 step %c1 {
    %q = memref.get_global @Q : memref<8xf64>
    %v = memref.load %q[%i] : memref<8xf64>
    %w = arith.addf %v, %cst : f64
    %t = memref.get_global @T : memref<8xf64>
    memref.store %w, %t[%i] : memref<8xf64>
  }
  return
}

// After LICM + CSE each global has exactly one SSA value in the function, so
// the flow dependence between the two nests is finally visible to the analysis.
// CHECK-LABEL: func.func @two_equations
// CHECK: memref.get_global @T
// CHECK: memref.get_global @Q
// CHECK-NOT: memref.get_global
