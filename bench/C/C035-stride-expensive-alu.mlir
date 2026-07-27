// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-partial-remat dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// C035: Partial remat — expensive ALU rejects on cost.
// Leaf at constant index (stride 0 → near-free at loop site), but the
// ALU chain is expensive (sqrt+div = 40 cycles).
// Consumer load on a small buffer (L1-fit, keep=1 stride-scaled).
// alu(40) + leaf(4) >= keep(1) → REJECT_COST.

module {
  func.func @stride_expensive_alu() -> f32 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f32
    %eps = arith.constant 0.001 : f32

    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %src = memref.alloc() : memref<1048576xf32>
    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %dst = memref.alloc() : memref<1024xf32>
    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %out = memref.alloc() : memref<1024xf32>

    // Writer: fills src.
    affine.for %i = 0 to 1048576 {
      %i32 = arith.index_cast %i : index to i32
      %f = arith.sitofp %i32 : i32 to f32
      affine.store %f, %src[%i] : memref<1048576xf32>
    }

    // Producer: expensive chain (sqrt+div), stored into small buffer.
    affine.for %k = 0 to 1024 {
      // expected-remark @below {{full-remat: REJECT_UNSAFE}}
      // expected-remark @below {{load: SINGLE}}
      %vs = affine.load %src[0] : memref<1048576xf32>
      %sq = arith.mulf %vs, %vs : f32
      %sum = arith.addf %sq, %one : f32
      %root = math.sqrt %sum : f32
      %denom = arith.addf %vs, %eps : f32
      %val = arith.divf %root, %denom : f32
      affine.store %val, %dst[%k] : memref<1024xf32>
    }

    // Consumer: reads from dst (L1-fit, stride-1 → keep=1).
    // alu(40) + leaf(4) = 44 >= keep(1) → REJECT_COST.
    affine.for %j = 0 to 1024 {
      // expected-remark @below {{load: SINGLE}}
      // expected-remark @below {{full-remat: REJECT_UNSAFE}}
      // expected-remark @below {{partial-remat: REJECT_COST}}
      %v = affine.load %dst[%j] : memref<1024xf32>
      affine.store %v, %out[%j] : memref<1024xf32>
    }

    // expected-remark @below {{load: SINGLE}}
    // expected-remark @below {{full-remat: REJECT_UNSAFE}}
    // expected-remark @below {{partial-remat: REJECT_COST}}
    %result = memref.load %out[%c0] : memref<1024xf32>
    memref.dealloc %src : memref<1048576xf32>
    memref.dealloc %dst : memref<1024xf32>
    memref.dealloc %out : memref<1024xf32>
    return %result : f32
  }
}

// The consumer loop load survives (REJECT_COST).
// CHECK-LABEL: func.func @stride_expensive_alu
// CHECK:         affine.for
// CHECK:           affine.load %{{.*}}[%{{.*}}] : memref<1024xf32>
