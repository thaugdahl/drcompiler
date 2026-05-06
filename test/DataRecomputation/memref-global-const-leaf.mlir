// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-partial-remat dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// Regression: a constant `memref.global` reachable via `memref.get_global`
// must end up folded — either by ConstantGlobalFold or by downstream remat.
// This guards the path that recognizes constant globals as alloc roots
// alongside `krnl.global`.

module {
  memref.global "private" constant @gconst : memref<4xf32> = dense<3.0>

  func.func @test() -> f32 {
    %c0 = arith.constant 0 : index
    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %dst = memref.alloc() : memref<1048576xf32>

    %k = memref.get_global @gconst : memref<4xf32>

    affine.for %j = 0 to 1048576 {
      // expected-remark @below {{constant-global-fold: ACCEPT}}
      %g = affine.load %k[0] : memref<4xf32>
      affine.store %g, %dst[%j] : memref<1048576xf32>
    }

    // expected-remark @below {{load: SINGLE}}
    // expected-remark @below {{full-remat: ACCEPT}}
    %out = affine.load %dst[%c0] : memref<1048576xf32>
    memref.dealloc %dst : memref<1048576xf32>
    return %out : f32
  }
}

// CHECK-LABEL: func.func @test
// CHECK:         affine.for
// CHECK:           arith.constant {{.*}} : f32
// CHECK:           affine.store
// CHECK:         }
// CHECK:         arith.constant
// CHECK:         return
