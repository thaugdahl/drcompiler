// RUN: dr-opt %s --allow-unregistered-dialect --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-buffer-elim=true dr-summary=true})' 2>&1 | FileCheck %s

// E006: Multiple loads share the same computation tree (all stores write
// the same constant). distinctTrees = 1 → shared subexpr discount makes
// elimCost = perElemCost * distinctTrees = 0 * 1 = 0.

// CHECK: buffer-elim {{.*}} loads=3{{.*}}elim=0
func.func @shared_subexpr() {
  %w = memref.alloca() : memref<3xi32>
  %c = arith.constant 42 : i32
  affine.store %c, %w[0] : memref<3xi32>
  affine.store %c, %w[1] : memref<3xi32>
  affine.store %c, %w[2] : memref<3xi32>
  %v0 = affine.load %w[0] : memref<3xi32>
  %v1 = affine.load %w[1] : memref<3xi32>
  %v2 = affine.load %w[2] : memref<3xi32>
  "use.consume"(%v0, %v1, %v2) : (i32, i32, i32) -> ()
  return
}
