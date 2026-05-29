// RUN: dr-opt %s --pass-pipeline='builtin.module(dr-affine-loop-fusion{arch-handler=x86-64-avx2})' | FileCheck %s

// Pick the x86-64-avx2 handler explicitly via CLI.  ArchHandler defaults
// (vec=16, fp=16) are well above the trivial pressure of this case, so the
// unified path produces the same fusion as upstream.  Confirms the
// arch-handler CLI plumbs end-to-end through the fork.

// CHECK-LABEL: func.func @avx2_smoke
// CHECK:         affine.for %{{.*}} = 0 to 32
// CHECK-NOT:     affine.for
// CHECK:         return

func.func @avx2_smoke(%A: memref<32xf32>, %B: memref<32xf32>) {
  %t = memref.alloc() : memref<32xf32>
  %c = arith.constant 2.0 : f32
  affine.for %i = 0 to 32 {
    %v = affine.load %A[%i] : memref<32xf32>
    %r = arith.mulf %v, %c : f32
    affine.store %r, %t[%i] : memref<32xf32>
  }
  affine.for %i = 0 to 32 {
    %v = affine.load %t[%i] : memref<32xf32>
    %r = arith.addf %v, %c : f32
    affine.store %r, %B[%i] : memref<32xf32>
  }
  memref.dealloc %t : memref<32xf32>
  return
}
