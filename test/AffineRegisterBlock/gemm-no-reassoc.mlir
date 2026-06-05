// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block))' | FileCheck %s

// Broadcast family (C=A*B): the second multiplicand B[k][j] is stride-1 in the
// spatial j, so family auto-selection keeps the wide mr x nr tile and emits NO
// fastmath -- LLVM SLP-vectorizes the independent j-lanes (each a distinct
// C[i][j]), which needs no FP reassociation and is RUINED by it (it would flip
// the backend to a horizontal-sum reduction strategy).  So no fastmath<fast>
// must appear.

module {
  func.func @gemm(%A: memref<64x64xf64>, %B: memref<64x64xf64>, %C: memref<64x64xf64>) {
    affine.for %i = 0 to 64 {
      affine.for %k = 0 to 64 {
        affine.for %j = 0 to 64 {
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

// Register-blocking fired (iter_args present) but NO reassociation was set.
// CHECK-LABEL: func.func @gemm
// CHECK: iter_args
// CHECK-NOT: fastmath<fast>
