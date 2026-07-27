// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission{test-diagnostics})' -verify-diagnostics | FileCheck %s
//
// D008: two structurally identical chains with different SSA names → FISSION.
// The fingerprint abstracts over SSA names: both loops produce
// sqrt(addf(mulf(LOAD, LOAD), C(1.0))).
// Chain: mulf(3) + addf(1) + sqrt(20) = 24.
// recompute = 2*24 = 48, keep = 24 + 1 + 2*4 = 33 → FISSION.

module {
  func.func @run(%src: memref<?xf64>, %count: index) -> f64 {
    %zero = arith.constant 0.0 : f64
    %one  = arith.constant 1.0 : f64

    // expected-remark @below {{memory-fission: FISSION (compute=24, consumers=2)}}
    // expected-remark @below {{materialized buffer for 2 consumers}}
    %total = affine.for %idx = 0 to %count iter_args(%running = %zero) -> (f64) {
      %elem = affine.load %src[%idx] : memref<?xf64>
      %prod = arith.mulf %elem, %elem : f64
      %inc  = arith.addf %prod, %one : f64
      %root = math.sqrt %inc : f64
      %next = arith.addf %running, %root : f64
      affine.yield %next : f64
    }

    %peak = affine.for %j = 0 to %count iter_args(%hi = %zero) -> (f64) {
      %val  = affine.load %src[%j] : memref<?xf64>
      %sqr  = arith.mulf %val, %val : f64
      %sum  = arith.addf %sqr, %one : f64
      %mag  = math.sqrt %sum : f64
      %bigger = arith.cmpf ogt, %mag, %hi : f64
      %next = arith.select %bigger, %mag, %hi : f64
      affine.yield %next : f64
    }

    %result = arith.addf %total, %peak : f64
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
