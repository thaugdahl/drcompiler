// RUN: dr-opt %s --pass-pipeline="builtin.module(dr-affine-loop-fusion{cpu-cost-model-file=%S/Inputs/avx2-fp6-realistic.json})" | FileCheck %s
// RUN: dr-opt %s --affine-loop-fusion | FileCheck %s --check-prefix=UPSTREAM

// Stronger version of decision-flip-tight-budget.mlir.  This time the
// unified cost model uses *realistic equal weights* (1, 1, 1) — the only
// non-default is fp_budget=6 (vs AVX2's 16) — and still rejects fusion.
// Made possible by `collectSliceOps`, which walks the producer's full
// SSA tree so RPA sees the 10-deep addf chain as added live values in
// the fused body rather than just the lone store.
//
// This is the headline target-aware-cost-model evidence: same program,
// different target FP register count -> different decision.

// CHECK-LABEL: func.func @pressure_flip
// drcomp under fp_budget=6 rejects: producer + consumer remain separate.
// CHECK: affine.for
// CHECK: affine.for
// CHECK: return

// UPSTREAM-LABEL: func.func @pressure_flip
// Upstream fuses regardless of target register file.
// UPSTREAM: affine.for
// UPSTREAM-NOT: affine.for
// UPSTREAM: return

func.func @pressure_flip(%A: memref<8xf32>, %B: memref<8xf32>) {
  %t = memref.alloc() : memref<8xf32>
  %c0 = arith.constant 0.0 : f32
  affine.for %i = 0 to 8 {
    %v = affine.load %A[%i] : memref<8xf32>
    %s0 = arith.addf %v, %c0 : f32
    %s1 = arith.addf %s0, %v : f32
    %s2 = arith.addf %s1, %v : f32
    %s3 = arith.addf %s2, %v : f32
    %s4 = arith.addf %s3, %v : f32
    %s5 = arith.addf %s4, %v : f32
    %s6 = arith.addf %s5, %v : f32
    %s7 = arith.addf %s6, %v : f32
    %s8 = arith.addf %s7, %v : f32
    %s9 = arith.addf %s8, %v : f32
    affine.store %s9, %t[%i] : memref<8xf32>
  }
  affine.for %i = 0 to 8 {
    %v = affine.load %t[%i] : memref<8xf32>
    %r = arith.mulf %v, %v : f32
    affine.store %r, %B[%i] : memref<8xf32>
  }
  memref.dealloc %t : memref<8xf32>
  return
}
