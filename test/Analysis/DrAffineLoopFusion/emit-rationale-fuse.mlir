// RUN: dr-opt %s --pass-pipeline="builtin.module(dr-affine-loop-fusion{emit-rationale=true})" -verify-diagnostics

// Companion to emit-rationale-reject.mlir.  On a small program with default
// budgets the unified path picks a fusion depth and reports it with the
// per-aspect breakdown.

func.func @producer_consumer(%A: memref<10xf32>, %B: memref<10xf32>) {
  %t = memref.alloc() : memref<10xf32>
  %c = arith.constant 1.0 : f32
  // expected-remark @below {{fusion-rationale: FUSE depth=1 fused_total=64 unfused_total=76}}
  affine.for %i = 0 to 10 {
    %v = affine.load %A[%i] : memref<10xf32>
    %r = arith.mulf %v, %c : f32
    affine.store %r, %t[%i] : memref<10xf32>
  }
  affine.for %i = 0 to 10 {
    %v = affine.load %t[%i] : memref<10xf32>
    %r = arith.addf %v, %c : f32
    affine.store %r, %B[%i] : memref<10xf32>
  }
  memref.dealloc %t : memref<10xf32>
  return
}
