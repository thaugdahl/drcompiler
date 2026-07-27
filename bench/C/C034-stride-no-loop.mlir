// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-partial-remat dr-test-diagnostics})' 2>&1 | FileCheck %s

// C034: Partial remat — consumer outside loop, leaf IV not live.
// Inner loop load gets partial-remat ACCEPT. Outer load (outside loop)
// gets REJECT_UNSAFE because the loop IV is not live at the use site.

module {
  func.func @stride_no_loop() -> f32 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f32

    %src = memref.alloc() : memref<1048576xf32>
    %dst = memref.alloc() : memref<1048576xf32>

    affine.for %i = 0 to 1048576 {
      %i32 = arith.index_cast %i : index to i32
      %f = arith.sitofp %i32 : i32 to f32
      affine.store %f, %src[%i] : memref<1048576xf32>
    }

    affine.for %k = 0 to 1048576 {
      %vs = affine.load %src[%k] : memref<1048576xf32>
      %add = arith.addf %vs, %one : f32
      affine.store %add, %dst[%k] : memref<1048576xf32>
    }

    %result = affine.load %dst[%c0] : memref<1048576xf32>
    memref.dealloc %src : memref<1048576xf32>
    memref.dealloc %dst : memref<1048576xf32>
    return %result : f32
  }
}

// Outer load processed first: REJECT_UNSAFE (IV not live).
// CHECK: partial-remat: REJECT_UNSAFE (reason=index-not-live)
// Inner loop load: ACCEPT.
// CHECK: partial-remat: ACCEPT
