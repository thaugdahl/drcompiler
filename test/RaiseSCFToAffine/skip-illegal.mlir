// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-raise-scf-to-affine))' | FileCheck %s
// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-raise-scf-to-affine{emit-rationale=true}))' -verify-diagnostics -o /dev/null

// Everything the legality check rejects stays scf.for.  Leaving part of a nest
// behind is safe: affine.for inside scf.for is valid IR and the affine
// analyses treat the scf parent conservatively -- the outer loop below raises
// while the inner one does not.

// A bound loaded at run time is neither a constant nor a valid affine symbol.
// CHECK-LABEL: func.func @dynamic_bound
// CHECK: affine.for
// CHECK:   scf.for
// expected-remark @below {{raised 1 scf.for, skipped 1}}
func.func @dynamic_bound(%A: memref<64xf64>, %N: memref<index>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.0 : f64
  scf.for %i = %c0 to %c1 step %c1 {
    %n = memref.load %N[] : memref<index>
    // expected-remark @below {{SKIP -- upper bound is neither a constant nor a valid affine symbol}}
    scf.for %j = %c0 to %n step %c1 {
      memref.store %cst, %A[%j] : memref<64xf64>
    }
  }
  return
}

// A loop-carried value is left alone rather than guessed at.
// CHECK-LABEL: func.func @iter_args
// CHECK: scf.for
// CHECK-NOT: affine.for
// expected-remark @below {{raised 0 scf.for, skipped 1}}
func.func @iter_args(%A: memref<64xf64>) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %zero = arith.constant 0.0 : f64
  // expected-remark @below {{SKIP -- loop carries iter_args}}
  %sum = scf.for %i = %c0 to %c64 step %c1 iter_args(%acc = %zero) -> (f64) {
    %v = memref.load %A[%i] : memref<64xf64>
    %s = arith.addf %acc, %v : f64
    scf.yield %s : f64
  }
  return %sum : f64
}

// A run-time step cannot become an affine.for step.
// CHECK-LABEL: func.func @dynamic_step
// CHECK: scf.for
// CHECK-NOT: affine.for
// expected-remark @below {{raised 0 scf.for, skipped 1}}
func.func @dynamic_step(%A: memref<64xf64>, %s: index) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 1.0 : f64
  // expected-remark @below {{SKIP -- step is not a constant}}
  scf.for %i = %c0 to %c64 step %s {
    memref.store %cst, %A[%i] : memref<64xf64>
  }
  return
}
