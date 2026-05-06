// RUN: dr-opt %s --allow-unregistered-dialect --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-partial-remat dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// `krnl.global` is the onnx-mlir constant-tensor op. dr-opt does not link
// the krnl dialect; we recognize it by op-name and treat it as a static,
// never-written global — so loads of its result are safe partial-remat
// leaves even though no `RootWriteSummary` entry exists.
//
// Pattern: in-loop kernel writes a krnl.global load into a large %dst
// buffer; downstream %out load triggers partial remat with the krnl.global
// load as the only leaf. The leaf must be ACCEPTed.

module {
  func.func @test() -> f32 {
    %c0 = arith.constant 0 : index
    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %dst = memref.alloc() : memref<1048576xf32>

    %k = "krnl.global"() {value = dense<3.0> : tensor<4xf32>} : () -> memref<4xf32>

    affine.for %j = 0 to 1048576 {
      %g = affine.load %k[0] : memref<4xf32>
      affine.store %g, %dst[%j] : memref<1048576xf32>
    }

    // expected-remark @below {{load: SINGLE}}
    // expected-remark @below {{full-remat: REJECT_UNSAFE}}
    // expected-remark @below {{partial-remat: ACCEPT}}
    %out = affine.load %dst[%c0] : memref<1048576xf32>
    memref.dealloc %dst : memref<1048576xf32>
    return %out : f32
  }
}

// After partial remat: the in-loop kernel is cloned at the consumer site.
// The cloned leaf is a load from %k (the krnl.global). Without const-global
// recognition this leaf would have been rejected with reason=intervening-write
// and partial-remat overall would have been REJECT_UNSAFE.
// CHECK-LABEL: func.func @test
// CHECK:         affine.for
// CHECK:         }
// CHECK:         %[[GLEAF:.*]] = affine.load %{{.*}}[0] : memref<4xf32>
// CHECK:         return %[[GLEAF]] : f32
