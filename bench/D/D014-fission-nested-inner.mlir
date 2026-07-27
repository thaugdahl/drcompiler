// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission{test-diagnostics})' -verify-diagnostics | FileCheck %s
//
// D014: chain in inner of 2-deep nest.  Two inner loops inside the same
// outer loop body share the same expensive computation from the same source.
// The inner loops have no further nesting, so hasInnerLoop=false.
// Chain: mulf(3) + addf(1) + sqrt(20) = 24.
// recompute = 2*24 = 48, keep = 24 + 1 + 2*4 = 33 → FISSION.

module {
  func.func @run(%x: memref<?xf64>, %n: index, %reps: index) -> f64 {
    %cst0 = arith.constant 0.0 : f64
    %cst1 = arith.constant 1.0 : f64

    %result = affine.for %r = 0 to %reps iter_args(%outer = %cst0) -> (f64) {
      // expected-remark @below {{memory-fission: FISSION (compute=24, consumers=2)}}
      // expected-remark @below {{materialized buffer for 2 consumers}}
      %sum = affine.for %i = 0 to %n iter_args(%acc = %cst0) -> (f64) {
        %xi = affine.load %x[%i] : memref<?xf64>
        %sq = arith.mulf %xi, %xi : f64
        %s  = arith.addf %sq, %cst1 : f64
        %v  = math.sqrt %s : f64
        %out = arith.addf %acc, %v : f64
        affine.yield %out : f64
      }

      %max = affine.for %i = 0 to %n iter_args(%best = %cst0) -> (f64) {
        %xi = affine.load %x[%i] : memref<?xf64>
        %sq = arith.mulf %xi, %xi : f64
        %s  = arith.addf %sq, %cst1 : f64
        %v  = math.sqrt %s : f64
        %gt = arith.cmpf ogt, %v, %best : f64
        %out = arith.select %gt, %v, %best : f64
        affine.yield %out : f64
      }

      %combined = arith.addf %sum, %max : f64
      %next = arith.addf %outer, %combined : f64
      affine.yield %next : f64
    }

    return %result : f64
  }
}

// CHECK-LABEL: func.func @run
// Outer REPS loop.
// CHECK:         affine.for
// Buffer alloc inside outer loop body.
// CHECK:           %[[BUF:.*]] = memref.alloc
// Producer loop.
// CHECK:           affine.for
// CHECK:             math.sqrt
// CHECK:             affine.store %{{.*}}, %[[BUF]]
// Consumer 1.
// CHECK:           affine.for
// CHECK:             affine.load %[[BUF]]
// CHECK-NOT:         math.sqrt
// CHECK:             affine.yield
// Consumer 2.
// CHECK:           affine.for
// CHECK:             affine.load %[[BUF]]
// CHECK-NOT:         math.sqrt
// CHECK:             affine.yield
// Dealloc inside outer loop body.
// CHECK:           memref.dealloc %[[BUF]]
