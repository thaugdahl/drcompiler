// RUN: dr-opt %s --allow-unregistered-dialect --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-partial-remat})' | FileCheck %s

// Phase C negative case: producer and consumer for-nests have
// MISMATCHED inner trip counts. matchEnclosingForNests must reject —
// the consumer load's `partial-remat` cannot fall back to IV-substitution,
// so the original `affine.load %alloc[...]` is preserved.

module {
  func.func @test() -> f32 {
    %k = "krnl.global"() {value = dense<3.0> : tensor<8192x128xf32>}
        : () -> memref<8192x128xf32>
    %ctrue = arith.constant true
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<1024x1024xf32>

    // Producer trip count: 1024 x 1024
    affine.for %p2 = 0 to 1024 {
      affine.for %p3 = 0 to 1024 {
        %row = arith.addi %p2, %c0 : index
        %row2 = arith.select %ctrue, %row, %c0 : index
        %v = memref.load %k[%row2, %p3] : memref<8192x128xf32>
        affine.store %v, %alloc[%p2, %p3] : memref<1024x1024xf32>
      }
    }

    // Consumer trip count differs on the inner loop (512 vs 1024).
    %sum = memref.alloc() : memref<f32>
    %z = arith.constant 0.0 : f32
    memref.store %z, %sum[] : memref<f32>
    affine.for %c2 = 0 to 1024 {
      affine.for %c3 = 0 to 512 {
        %x = affine.load %alloc[%c2, %c3] : memref<1024x1024xf32>
        %acc = memref.load %sum[] : memref<f32>
        %nx = arith.addf %acc, %x : f32
        memref.store %nx, %sum[] : memref<f32>
      }
    }

    %final = memref.load %sum[] : memref<f32>
    memref.dealloc %alloc : memref<1024x1024xf32>
    memref.dealloc %sum : memref<f32>
    return %final : f32
  }
}

// Mismatched IV nests: IV substitution is rejected, the consumer load
// of %alloc remains in the IR.
// CHECK-LABEL: func.func @test
// CHECK:         affine.for %{{.*}} = 0 to 1024
// CHECK:           affine.for %{{.*}} = 0 to 512
// CHECK:             affine.load %{{.*}} : memref<1024x1024xf32>
