// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission{test-diagnostics})' -verify-diagnostics | FileCheck %s
//
// D010: affine loop trip=256 (buffer 256*8=2KB, fits L1).
// Chain: mulf(3) + addf(1) + sqrt(20) = 24.
// recompute = 2*24 = 48, keep = 24 + 1 + 2*4 = 33 → FISSION.

module {
  func.func @run(%x: memref<256xf64>) -> f64 {
    %cst0 = arith.constant 0.0 : f64
    %cst1 = arith.constant 1.0 : f64

    // expected-remark @below {{memory-fission: FISSION (compute=24, consumers=2)}}
    // expected-remark @below {{materialized buffer for 2 consumers}}
    %sum = affine.for %i = 0 to 256 iter_args(%acc = %cst0) -> (f64) {
      %xi = affine.load %x[%i] : memref<256xf64>
      %sq = arith.mulf %xi, %xi : f64
      %s  = arith.addf %sq, %cst1 : f64
      %r  = math.sqrt %s : f64
      %out = arith.addf %acc, %r : f64
      affine.yield %out : f64
    }

    %max = affine.for %i = 0 to 256 iter_args(%best = %cst0) -> (f64) {
      %xi = affine.load %x[%i] : memref<256xf64>
      %sq = arith.mulf %xi, %xi : f64
      %s  = arith.addf %sq, %cst1 : f64
      %r  = math.sqrt %s : f64
      %gt = arith.cmpf ogt, %r, %best : f64
      %out = arith.select %gt, %r, %best : f64
      affine.yield %out : f64
    }

    %result = arith.addf %sum, %max : f64
    return %result : f64
  }
}

// CHECK-LABEL: func.func @run
// CHECK:         %[[BUF:.*]] = memref.alloc
// CHECK:         affine.for
// CHECK:           math.sqrt
// CHECK:           affine.store %{{.*}}, %[[BUF]]
// CHECK:         affine.for
// CHECK:           affine.load %[[BUF]]
// CHECK-NOT:       math.sqrt
// CHECK:           affine.yield
// CHECK:         affine.for
// CHECK:           affine.load %[[BUF]]
// CHECK-NOT:       math.sqrt
// CHECK:           affine.yield
// CHECK:         memref.dealloc %[[BUF]]
