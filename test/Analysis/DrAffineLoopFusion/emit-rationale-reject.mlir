// RUN: dr-opt %s --pass-pipeline="builtin.module(dr-affine-loop-fusion{cpu-cost-model-file=%S/Inputs/avx2-fp6-realistic.json emit-rationale=true})" -verify-diagnostics

// emit-rationale=true emits one fusion-rationale remark per fusion decision
// from the unified cost model.  Companion to decision-flip-realistic-weights.mlir;
// verifies that the breakdown (fused_total, unfused_total) is what the cost
// model is actually computing — the model is now auditable from lit.

func.func @pressure_flip(%A: memref<8xf32>, %B: memref<8xf32>) {
  %t = memref.alloc() : memref<8xf32>
  %c0 = arith.constant 0.0 : f32
  // expected-remark @below {{fusion-rationale: REJECT fused_total=164 >= unfused_total=128}}
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
