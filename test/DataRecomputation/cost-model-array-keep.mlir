// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// Array buffer with expensive computation (sqrt + div), 2 consumer functions.
// Buffer is 16KB (memref<2048xf64>), fits in L1.
// Cost model should KEEP the buffer rather than recompute 2×.
//
// Derived from: cgeist -O0 on separate producer/consumer functions
// operating on a shared double array.

module {
  func.func @fill(%w: memref<?xf64>, %x: memref<?xf64>, %n: i32) {
    %cst1 = arith.constant 1.0 : f64
    %eps = arith.constant 0.001 : f64
    %ni = arith.index_cast %n : i32 to index
    affine.for %i = 0 to %ni {
      %xi = affine.load %x[%i] : memref<?xf64>
      %sq = arith.mulf %xi, %xi : f64
      %s = arith.addf %sq, %cst1 : f64
      %r = math.sqrt %s : f64
      %d = arith.addf %xi, %eps : f64
      %v = arith.divf %r, %d : f64
      affine.store %v, %w[%i] : memref<?xf64>
    }
    return
  }

  func.func @reduce_sum(%w: memref<?xf64>, %n: i32) -> f64 {
    %cst0 = arith.constant 0.0 : f64
    %ni = arith.index_cast %n : i32 to index
    %s = affine.for %i = 0 to %ni iter_args(%acc = %cst0) -> (f64) {
      // expected-remark @below {{cost-model: SKIP_LOAD (buffer kept)}}
      // expected-remark @below {{interproc-cross: REJECT_COST}}
      // expected-remark @below {{load: SINGLE}}
      %v = affine.load %w[%i] : memref<?xf64>
      %r = arith.addf %acc, %v : f64
      affine.yield %r : f64
    }
    return %s : f64
  }

  func.func @reduce_max(%w: memref<?xf64>, %n: i32) -> f64 {
    %c0 = arith.constant 0 : index
    // expected-remark @below {{cost-model: SKIP_LOAD (buffer kept)}}
    // expected-remark @below {{interproc-cross: REJECT_COST}}
    // expected-remark @below {{load: SINGLE}}
    %init = affine.load %w[%c0] : memref<?xf64>
    %ni = arith.index_cast %n : i32 to index
    %m = affine.for %i = 1 to %ni iter_args(%best = %init) -> (f64) {
      // expected-remark @below {{cost-model: SKIP_LOAD (buffer kept)}}
      // expected-remark @below {{interproc-cross: REJECT_COST}}
      // expected-remark @below {{load: SINGLE}}
      %v = affine.load %w[%i] : memref<?xf64>
      %gt = arith.cmpf ogt, %v, %best : f64
      %r = arith.select %gt, %v, %best : f64
      affine.yield %r : f64
    }
    return %m : f64
  }

  func.func @run(%x: memref<?xf64>, %n: i32) -> f64 {
    // expected-remark @+1 {{cost-model: KEEP}}
    %buf = memref.alloc() : memref<2048xf64>
    %cast = memref.cast %buf : memref<2048xf64> to memref<?xf64>
    call @fill(%cast, %x, %n) : (memref<?xf64>, memref<?xf64>, i32) -> ()
    %s = call @reduce_sum(%cast, %n) : (memref<?xf64>, i32) -> f64
    %m = call @reduce_max(%cast, %n) : (memref<?xf64>, i32) -> f64
    %r = arith.addf %s, %m : f64
    memref.dealloc %buf : memref<2048xf64>
    return %r : f64
  }
}

// The affine.loads should survive — cost model says KEEP.
// CHECK-LABEL: func.func @reduce_sum
// CHECK:         affine.load
// CHECK-LABEL: func.func @reduce_max
// CHECK:         affine.load
