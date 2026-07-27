// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission{test-diagnostics})' -verify-diagnostics | FileCheck %s
//
// D024: single div (cost=15 exactly), no other ops in chain besides constant.
// Chain: divf(15) + constant(0) = 15.  Exactly at minChainCost.
// recompute = 2*15 = 30, keep = 15 + 1 + 2*4 = 24 → FISSION.
// Same cost as D022 but with a different chain structure (div by constant,
// no mulf/addf prefix).

module {
  func.func @run(%x: memref<?xf64>, %n: index) -> f64 {
    %cst0 = arith.constant 0.0 : f64
    %cst2 = arith.constant 2.0 : f64

    // expected-remark @below {{memory-fission: FISSION (compute=15, consumers=2)}}
    // expected-remark @below {{materialized buffer for 2 consumers}}
    %sum = affine.for %i = 0 to %n iter_args(%acc = %cst0) -> (f64) {
      %xi = affine.load %x[%i] : memref<?xf64>
      %v  = arith.divf %xi, %cst2 : f64
      %out = arith.addf %acc, %v : f64
      affine.yield %out : f64
    }

    %prod = affine.for %i = 0 to %n iter_args(%acc = %cst2) -> (f64) {
      %xi = affine.load %x[%i] : memref<?xf64>
      %v  = arith.divf %xi, %cst2 : f64
      %out = arith.mulf %acc, %v : f64
      affine.yield %out : f64
    }

    %result = arith.addf %sum, %prod : f64
    return %result : f64
  }
}

// CHECK-LABEL: func.func @run
// CHECK:         %[[BUF:.*]] = memref.alloc
// CHECK:         affine.for
// CHECK:           arith.divf
// CHECK:           affine.store %{{.*}}, %[[BUF]]
// CHECK:         affine.for
// CHECK:           affine.load %[[BUF]]
// CHECK-NOT:       arith.divf
// CHECK:           affine.yield
// CHECK:         affine.for
// CHECK:           affine.load %[[BUF]]
// CHECK-NOT:       arith.divf
// CHECK:           affine.yield
// CHECK:         memref.dealloc %[[BUF]]
