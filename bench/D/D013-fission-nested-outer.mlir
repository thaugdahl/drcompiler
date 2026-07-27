// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission{test-diagnostics})' -verify-diagnostics | FileCheck %s
//
// D013: chain in outer of 2-deep nest (outer loops contain inner loops).
// The expensive ops (sqrt) are in the outer loop body directly, and
// duplicated across 2 outer loops.  However, each outer loop body also
// contains an inner affine.for, triggering the hasInnerLoop safety check.
// Cost model says FISSION, but the transformation is skipped.
// Chain: mulf(3) + addf(1) + sqrt(20) = 24. consumers=2.
// recompute = 48, keep = 33 → FISSION remark emitted, but no transform.

module {
  func.func @run(%x: memref<?xf64>, %y: memref<?xf64>, %n: index) -> f64 {
    %cst0 = arith.constant 0.0 : f64
    %cst1 = arith.constant 1.0 : f64

    // expected-remark @below {{memory-fission: FISSION (compute=24, consumers=2)}}
    %sum = affine.for %i = 0 to %n iter_args(%acc = %cst0) -> (f64) {
      %xi = affine.load %x[%i] : memref<?xf64>
      %sq = arith.mulf %xi, %xi : f64
      %s  = arith.addf %sq, %cst1 : f64
      %r  = math.sqrt %s : f64
      // Inner loop that prevents fission of the outer loop.
      %inner = affine.for %j = 0 to %n iter_args(%iacc = %cst0) -> (f64) {
        %yj = affine.load %y[%j] : memref<?xf64>
        %p  = arith.mulf %r, %yj : f64
        %next = arith.addf %iacc, %p : f64
        affine.yield %next : f64
      }
      %out = arith.addf %acc, %inner : f64
      affine.yield %out : f64
    }

    %prod = affine.for %i = 0 to %n iter_args(%acc = %cst1) -> (f64) {
      %xi = affine.load %x[%i] : memref<?xf64>
      %sq = arith.mulf %xi, %xi : f64
      %s  = arith.addf %sq, %cst1 : f64
      %r  = math.sqrt %s : f64
      // Inner loop that prevents fission of the outer loop.
      %inner = affine.for %j = 0 to %n iter_args(%iacc = %cst1) -> (f64) {
        %yj = affine.load %y[%j] : memref<?xf64>
        %p  = arith.mulf %r, %yj : f64
        %next = arith.mulf %iacc, %p : f64
        affine.yield %next : f64
      }
      %out = arith.mulf %acc, %inner : f64
      affine.yield %out : f64
    }

    %result = arith.addf %sum, %prod : f64
    return %result : f64
  }
}

// CHECK-LABEL: func.func @run
// No fission — nested loops block the transformation.
// CHECK-NOT:     memref.alloc
// CHECK:         affine.for
// CHECK:           math.sqrt
// CHECK:           affine.for
// CHECK:         affine.for
// CHECK:           math.sqrt
// CHECK:           affine.for
