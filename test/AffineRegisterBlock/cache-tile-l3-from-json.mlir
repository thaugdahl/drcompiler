// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block{mr=2 nr=2 cache-tile=true mc=16 nc=16 kc=16 cpu-cost-model-file=%S/Inputs/small-l3.json}))' | FileCheck %s

// WP1 (COSTMODEL_V4_SPEC §2): register-block now sources l3-size from the
// cost-model JSON's `cache` block (no CLI l3-size given).  The fixture sets
// l3_size=16384, so the ~96 KiB 64x64 band exceeds the effective LLC and the
// cache-tiling path fires — exactly as it does when l3-size=16384 is passed on
// the CLI (see gemm-cache-tile.mlir).  This proves the new MachineModel wiring
// reaches the pass.

module {
  func.func @gemm(%A: memref<64x64xf64>, %B: memref<64x64xf64>, %C: memref<64x64xf64>) {
    affine.for %i = 0 to 64 {
      affine.for %j = 0 to 64 {
        affine.for %k = 0 to 64 {
          %a = affine.load %A[%i, %k] : memref<64x64xf64>
          %b = affine.load %B[%k, %j] : memref<64x64xf64>
          %c = affine.load %C[%i, %j] : memref<64x64xf64>
          %p = arith.mulf %a, %b : f64
          %s = arith.addf %c, %p : f64
          affine.store %s, %C[%i, %j] : memref<64x64xf64>
        }
      }
    }
    return
  }
}

// Cache-tile loops stepped by the tile size (16) — present only if the JSON
// l3-size override drove the gate.
// CHECK: affine.for %{{.*}} = 0 to 64 step 16
// CHECK:   affine.for %{{.*}} = 0 to 64 step 16
// CHECK:     affine.for %{{.*}} = 0 to 64 step 16
// CHECK:       affine.for %{{.*}} step 8
// CHECK:         affine.vector_load %{{.*}} : memref<64x64xf64>, vector<8xf64>
