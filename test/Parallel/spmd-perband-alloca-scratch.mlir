// RUN: dr-opt %s --pass-pipeline='builtin.module(dr-par-bubbles{par-test-diagnostics})' 2>&1 | FileCheck %s --check-prefix=AXIS
// RUN: dr-opt %s --pass-pipeline='builtin.module(dr-par-bubbles{par-spmd-perband})' | FileCheck %s

// A GEMM+bias band in the demote/promote form onnx-mlir produces: an inner
// iter_arg reduction (the K loop) PLUS a residual per-output `memref.alloca`
// scalar accumulator (used to thread the init in and add the bias out).  That
// alloca is loop-PRIVATE scratch -- a fresh allocation per (m,n) iteration that
// never escapes the band -- so the m and n output axes are parallel.
//
// Regression guard for the ParAliasOracle fix: previously the alloca's
// allocation effect + its scalar load/store made axisConflict return
// Unknown/Carried, marking m and n SEQUENTIAL(conservative) -> the whole GEMM
// fell to par.critical (serial).  This serialized openai-gpt's QKV/FC GEMMs
// (the bulk of the FLOPs); privatizing the in-loop alloca recovers them as
// par.forall and lets the transformer scale (PARALLEL_SPMD_SPEC.md §11.13).

// AXIS: par-bubble axis: SEQUENTIAL (reduction)
// AXIS: par-bubble axis: PARALLEL
// AXIS: par-bubble axis: PARALLEL

// CHECK-LABEL: func.func @gemm_bias
// CHECK:         par.region {
// CHECK:           par.forall([0], [128], [1]) {
// CHECK-NOT:       par.critical
// CHECK:         }
func.func @gemm_bias(%in: memref<128x768xf32>, %wt: memref<768x2304xf32>,
                     %bias: memref<2304xf32>, %out: memref<128x2304xf32>) {
  %cst = arith.constant 0.0 : f32
  affine.for %m = 0 to 128 {
    affine.for %n = 0 to 2304 {
      %alloca = memref.alloca() : memref<f32>
      affine.store %cst, %alloca[] : memref<f32>
      %i = affine.load %alloca[] : memref<f32>
      %acc = affine.for %k = 0 to 768 iter_args(%a = %i) -> (f32) {
        %x = affine.load %in[%m, %k] : memref<128x768xf32>
        %w = affine.load %wt[%k, %n] : memref<768x2304xf32>
        %p = arith.mulf %x, %w : f32
        %s = arith.addf %p, %a : f32
        affine.yield %s : f32
      }
      affine.store %acc, %alloca[] : memref<f32>
      %r = affine.load %alloca[] : memref<f32>
      %b = affine.load %bias[%n] : memref<2304xf32>
      %o = arith.addf %r, %b : f32
      affine.store %o, %out[%m, %n] : memref<128x2304xf32>
    }
  }
  return
}
