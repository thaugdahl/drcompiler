// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-cost-model dr-buffer-elim=true dr-summary=true})' 2>&1 | FileCheck %s --check-prefix=NOPROMAT
// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-cost-model dr-partial-remat dr-buffer-elim=true dr-summary=true})' 2>&1 | FileCheck %s --check-prefix=PROMAT

// G016: Interaction between partial remat and buffer elim analysis.
// Both configurations show buffer-elim feasibility reports.
// The test verifies both pipelines complete and produce summaries.

module {
  func.func @elim_needs_partial() -> f32 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f32

    %src = memref.alloc() : memref<1048576xf32>
    %dst = memref.alloc() : memref<1048576xf32>

    // IV-dependent writer to %src (breaks chain remat of leaf).
    affine.for %i = 0 to 1048576 {
      %ic = arith.index_cast %i : index to i32
      %f = arith.sitofp %ic : i32 to f32
      affine.store %f, %src[%i] : memref<1048576xf32>
    }

    // %dst writer: constant-index leaf load of %src + add.
    affine.for %k = 0 to 65536 {
      %vs = affine.load %src[0] : memref<1048576xf32>
      %add = arith.addf %vs, %one : f32
      affine.store %add, %dst[%k * 16] : memref<1048576xf32>
    }

    // Consumer reads %dst.
    %result = memref.load %dst[%c0] : memref<1048576xf32>
    memref.dealloc %src : memref<1048576xf32>
    memref.dealloc %dst : memref<1048576xf32>
    return %result : f32
  }
}

// Without partial remat: buffer elim reports on both src and dst.
// NOPROMAT: buffer-elim
// NOPROMAT: buffer-elim

// With partial remat: also reports on both, pipeline completes.
// PROMAT: buffer-elim
// PROMAT: buffer-elim
