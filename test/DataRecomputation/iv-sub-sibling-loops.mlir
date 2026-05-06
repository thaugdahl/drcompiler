// RUN: dr-opt %s --allow-unregistered-dialect --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-partial-remat})' | FileCheck %s

// Phase C: IV-substituted partial remat across sibling for-nests with
// matching iteration domain. The leaf load reads from a krnl.global
// constant, with an index-computing chain (arith.addi / arith.select)
// over the producer IVs. The consumer reads %alloc at the same affine
// indices. The IV mapping (producer p2/p3 ↦ consumer c2/c3) lets the
// leaf load and its index chain clone at the consumer site with
// consumer IVs substituted for producer IVs.

module {
  func.func @test() -> f32 {
    %k = "krnl.global"() {value = dense<3.0> : tensor<8192x128xf32>}
        : () -> memref<8192x128xf32>
    %ctrue = arith.constant true
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<1024x1024xf32>

    affine.for %p2 = 0 to 1024 {
      affine.for %p3 = 0 to 1024 {
        %row = arith.addi %p2, %c0 : index
        %row2 = arith.select %ctrue, %row, %c0 : index
        %v = memref.load %k[%row2, %p3] : memref<8192x128xf32>
        affine.store %v, %alloc[%p2, %p3] : memref<1024x1024xf32>
      }
    }

    %sum = memref.alloc() : memref<f32>
    %z = arith.constant 0.0 : f32
    memref.store %z, %sum[] : memref<f32>
    affine.for %c2 = 0 to 1024 {
      affine.for %c3 = 0 to 1024 {
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

// After IV-substituted partial remat, the consumer body contains a
// CLONED leaf load of %k and the cloned index chain (arith.addi /
// arith.select) using the consumer IVs in place of producer IVs. The
// original consumer load of %alloc is replaced by the clone's result.
//
// CHECK-LABEL: func.func @test
// CHECK:         %[[K:.*]] = "krnl.global"
// CHECK:         affine.for %[[P2:.*]] = 0 to 1024
// CHECK:           affine.for %[[P3:.*]] = 0 to 1024
// CHECK:             affine.store
// CHECK:         affine.for %[[C2:.*]] = 0 to 1024
// CHECK:           affine.for %[[C3:.*]] = 0 to 1024
// CHECK:             arith.addi %[[C2]]
// CHECK:             arith.select
// CHECK:             memref.load %[[K]]{{.*}} memref<8192x128xf32>
