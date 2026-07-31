// The recompute-cone THROUGHPUT floor divides by the target's issue width,
// which is resolved (CLI handler > JSON > handler default) instead of the
// `kIssueWidth = 4` literal it used to be.
//
// The stored value is deliberately WIDE and SHALLOW: 8 independent sqrt (20
// cycles each) reduced by a 3-deep addf tree (1 each).  So
//   totalCost    = 8*20 + 7*1 = 167
//   criticalPath = 20 + 3      = 23
// and estimateComputeCost returns max(criticalPath, ceil(totalCost/width)):
//   width 1 -> 167 (no superscalar overlap)
//   width 4 -> 42  (the historical hardcoded value; the no-JSON default)
//   width 8 -> 23  (apple-m-series; the critical path now binds)
// A dependent chain would be critical-path-bound and show no width sensitivity
// at all — that asymmetry is the point of the parameter.

// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics cpu-cost-model-file=%S/Inputs/issue-width-1.json})' -o /dev/null 2>&1 | FileCheck %s --check-prefix=NARROW
// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics cpu-cost-model-file=%S/Inputs/apple-m-handler.json})' -o /dev/null 2>&1 | FileCheck %s --check-prefix=WIDE
// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics dr-arch-handler=apple-m-series})' -o /dev/null 2>&1 | FileCheck %s --check-prefix=WIDE

// DEFAULT: cost-model: {{.*}}compute=42
// NARROW: cost-model: {{.*}}compute=167
// WIDE: cost-model: {{.*}}compute=23

module {
  func.func @wide(%src: memref<64xf32>) -> f32 {
    %c0 = arith.constant 0 : index
    %dst = memref.alloc() : memref<8xf32>
    affine.for %i = 0 to 8 {
      %a = affine.load %src[%i] : memref<64xf32>
      %b = affine.load %src[%i + 8] : memref<64xf32>
      %c = affine.load %src[%i + 16] : memref<64xf32>
      %d = affine.load %src[%i + 24] : memref<64xf32>
      %e = affine.load %src[%i + 32] : memref<64xf32>
      %f = affine.load %src[%i + 40] : memref<64xf32>
      %g = affine.load %src[%i + 48] : memref<64xf32>
      %h = affine.load %src[%i + 56] : memref<64xf32>
      %sa = math.sqrt %a : f32
      %sb = math.sqrt %b : f32
      %sc = math.sqrt %c : f32
      %sd = math.sqrt %d : f32
      %se = math.sqrt %e : f32
      %sf = math.sqrt %f : f32
      %sg = math.sqrt %g : f32
      %sh = math.sqrt %h : f32
      %p1 = arith.addf %sa, %sb : f32
      %p2 = arith.addf %sc, %sd : f32
      %p3 = arith.addf %se, %sf : f32
      %p4 = arith.addf %sg, %sh : f32
      %q1 = arith.addf %p1, %p2 : f32
      %q2 = arith.addf %p3, %p4 : f32
      %r  = arith.addf %q1, %q2 : f32
      affine.store %r, %dst[%i] : memref<8xf32>
    }
    %out = affine.load %dst[%c0] : memref<8xf32>
    memref.dealloc %dst : memref<8xf32>
    return %out : f32
  }
}
