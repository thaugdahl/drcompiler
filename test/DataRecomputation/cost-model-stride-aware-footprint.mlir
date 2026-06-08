// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-footprint-analysis dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// Stride-aware intervening footprint (spike #3).
//
// Two kernels are identical except for the access STRIDE of the intervening
// loop between the single store and its two loads. Both loops run the SAME
// number of iterations (4096); they differ only in whether the access is a
// contiguous stream or a column gather.
//
// The footprint estimator now charges the cache-line-amortized footprint per
// iteration: a contiguous (stride-1) stream contributes one element/iter, but
// a strided/gather access that skips whole lines contributes up to a full
// 64-byte line/iter. Coverage-gated and monotone-upward: stride<=1 (and
// unknown stride) keep the old per-element estimate, so @stream below is
// exactly what the pre-spike, stride-blind model computed for BOTH kernels.
//
//   @stream  (stride 1):  4096 * 4 B  = 16 KB  -> buffer stays in L1 (4 cy)
//   @gather  (stride 16): 4096 * 64 B = 256 KB -> buffer evicted to L2 (12 cy)
//
// With cheap-enough access (sqrt, compute=20, 2 consumers) the L1-resident
// case favors KEEP, but the gather-evicted case crosses into L2 and flips the
// decision to RECOMPUTE. The pre-spike model priced both as L1 and KEPT both.

// CHECK-LABEL: func.func @stream
// CHECK:         memref.load
// CHECK:         memref.load
func.func @stream(%x: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %ci = arith.constant 1 : i32
  // expected-remark @below {{cost-model: KEEP}}
  %buf = memref.alloc() : memref<1xf32>
  %big = memref.alloc() : memref<4096xi32>
  %val = math.sqrt %x : f32
  memref.store %val, %buf[%c0] : memref<1xf32>
  affine.for %i = 0 to 4096 {
    affine.store %ci, %big[%i] : memref<4096xi32>
  }
  // expected-remark @below {{load: SINGLE}}
  // expected-remark @below {{cost-model: SKIP_LOAD}}
  %a = memref.load %buf[%c0] : memref<1xf32>
  // expected-remark @below {{load: SINGLE}}
  // expected-remark @below {{cost-model: SKIP_LOAD}}
  %b = memref.load %buf[%c0] : memref<1xf32>
  %r = arith.addf %a, %b : f32
  memref.dealloc %big : memref<4096xi32>
  memref.dealloc %buf : memref<1xf32>
  return %r : f32
}

// CHECK-LABEL: func.func @gather
// CHECK-NOT:     memref.load %{{.*}} : memref<1xf32>
// CHECK:         return
func.func @gather(%x: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %ci = arith.constant 1 : i32
  // expected-remark @below {{cost-model: RECOMPUTE}}
  %buf = memref.alloc() : memref<1xf32>
  %big = memref.alloc() : memref<4096x16xi32>
  %val = math.sqrt %x : f32
  memref.store %val, %buf[%c0] : memref<1xf32>
  affine.for %i = 0 to 4096 {
    affine.store %ci, %big[%i, 0] : memref<4096x16xi32>
  }
  // expected-remark @below {{load: SINGLE}}
  // expected-remark @below {{direct-forward: ACCEPT}}
  %a = memref.load %buf[%c0] : memref<1xf32>
  // expected-remark @below {{load: SINGLE}}
  // expected-remark @below {{direct-forward: ACCEPT}}
  %b = memref.load %buf[%c0] : memref<1xf32>
  %r = arith.addf %a, %b : f32
  memref.dealloc %big : memref<4096x16xi32>
  memref.dealloc %buf : memref<1xf32>
  return %r : f32
}
