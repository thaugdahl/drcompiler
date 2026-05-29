// RUN: dr-opt %s --pass-pipeline="builtin.module(dr-affine-loop-fusion{cpu-cost-model-file=%S/Inputs/reject-tight-budget.json})" | FileCheck %s
// RUN: dr-opt %s --affine-loop-fusion | FileCheck %s --check-prefix=UPSTREAM

// The producer-consumer pair fuses cleanly under upstream's placeholder cost
// model.  Under the unified cost model with tight register budgets and a
// large beta_reg weight, the spill penalty for the fused candidate exceeds
// the unfused baseline and fusion is rejected.

// This is the headline empirical evidence that the unified cost model has
// signal — its decision diverges from upstream's on the same program.

// CHECK-LABEL: func.func @producer_consumer
// drcomp rejects fusion: two separate affine.for loops remain.
// CHECK: affine.for
// CHECK: affine.for
// CHECK: return

// UPSTREAM-LABEL: func.func @producer_consumer
// Upstream fuses: a single affine.for.
// UPSTREAM: affine.for
// UPSTREAM-NOT: affine.for
// UPSTREAM: return

func.func @producer_consumer(%A: memref<10xf32>, %B: memref<10xf32>) {
  %t = memref.alloc() : memref<10xf32>
  %c = arith.constant 1.0 : f32
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
