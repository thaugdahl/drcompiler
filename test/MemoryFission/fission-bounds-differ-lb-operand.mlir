// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission{test-diagnostics})' -verify-diagnostics | FileCheck %s

// S2-B: Two loops share an expensive computation chain (sqrt + div) and have
// the same affine bound maps, but their LOWER BOUND OPERANDS differ
// (%lb1 vs %lb2).  Without the fix, the bounds-match check passed because
// it only compared maps + upper-bound operands.  Fission would then create a
// producer loop using firstLoop's lower bound (%lb1) and indexing into the
// buffer assuming index 0 corresponds to %lb1 -- but the second consumer
// starts at %lb2 and would read garbage / out-of-buffer memory.
//
// With the fix, the bounds-match check rejects this candidate group.  The
// cost-model still emits FISSION (the group was identified), but the
// transformation is skipped: no producer loop, no buffer alloc, original
// loops untouched.

module {
  func.func @run(%x: memref<?xf64>, %n: index, %lb1: index, %lb2: index)
      -> f64 {
    %cst0 = arith.constant 0.0 : f64
    %cst1 = arith.constant 1.0 : f64
    %eps  = arith.constant 0.001 : f64

    // expected-remark @below {{memory-fission: FISSION}}
    %sum = affine.for %i = affine_map<()[s0] -> (s0)>()[%lb1] to %n
        iter_args(%acc = %cst0) -> (f64) {
      %xi = affine.load %x[%i] : memref<?xf64>
      %sq = arith.mulf %xi, %xi : f64
      %s  = arith.addf %sq, %cst1 : f64
      %r  = math.sqrt %s : f64
      %d  = arith.addf %xi, %eps : f64
      %v  = arith.divf %r, %d : f64
      %out = arith.addf %acc, %v : f64
      affine.yield %out : f64
    }

    %max = affine.for %i = affine_map<()[s0] -> (s0)>()[%lb2] to %n
        iter_args(%best = %cst0) -> (f64) {
      %xi = affine.load %x[%i] : memref<?xf64>
      %sq = arith.mulf %xi, %xi : f64
      %s  = arith.addf %sq, %cst1 : f64
      %r  = math.sqrt %s : f64
      %d  = arith.addf %xi, %eps : f64
      %v  = arith.divf %r, %d : f64
      %gt = arith.cmpf ogt, %v, %best : f64
      %out = arith.select %gt, %v, %best : f64
      affine.yield %out : f64
    }

    %result = arith.addf %sum, %max : f64
    return %result : f64
  }
}

// CHECK-LABEL: func.func @run
// No producer loop, no buffer allocation -- transformation rejected.
// CHECK-NOT: memref.alloc
// Both consumer loops still contain the original computation.
// CHECK:     affine.for
// CHECK:       math.sqrt
// CHECK:       arith.divf
// CHECK:     affine.for
// CHECK:       math.sqrt
// CHECK:       arith.divf
