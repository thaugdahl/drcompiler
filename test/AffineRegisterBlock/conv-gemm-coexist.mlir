// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block{mr=8 nr=16 vl=16}))' | FileCheck %s

// A function containing BOTH a direct-conv band (ic/kh/kw) and a 1x1-conv GEMM
// must vectorize each on its own path: the conv band via the direct-conv
// kernel, the GEMM via the broadcast kernel.  Regression guard for
// findReductionLoopUnder returning a conv-band inner (kw) loop and derailing the
// GEMM (the 1x1s silently fell back to scalar once 3x3 convs were de-promoted).

// CHECK-LABEL: func.func @both
func.func @both(%in: memref<64x58x66xf32>, %wc: memref<64x64x3x3xf32>, %Yc: memref<64x56x64xf32>,
                %A: memref<64x64xf32>, %B: memref<64x256xf32>, %Yg: memref<64x256xf32>) {
  // --- interior 3x3 conv band ---
  affine.for %oc = 0 to 64 {
    affine.for %oh = 0 to 56 {
      affine.for %ow = 0 to 64 {
        affine.for %ic = 0 to 64 {
          affine.for %kh = 0 to 3 {
            affine.for %kw = 0 to 3 {
              %i = affine.load %in[%ic, %oh + %kh, %ow + %kw] : memref<64x58x66xf32>
              %w = affine.load %wc[%oc, %ic, %kh, %kw] : memref<64x64x3x3xf32>
              %c = affine.load %Yc[%oc, %oh, %ow] : memref<64x56x64xf32>
              %p = arith.mulf %i, %w : f32
              %s = arith.addf %c, %p : f32
              affine.store %s, %Yc[%oc, %oh, %ow] : memref<64x56x64xf32>
            }
          }
        }
      }
    }
  }
  // --- 1x1 GEMM: Yg[m,n] += A[m,k]*B[k,n] ---
  affine.for %m = 0 to 64 {
    affine.for %n = 0 to 256 {
      affine.for %k = 0 to 64 {
        %a = affine.load %A[%m, %k] : memref<64x64xf32>
        %b = affine.load %B[%k, %n] : memref<64x256xf32>
        %c = affine.load %Yg[%m, %n] : memref<64x256xf32>
        %p = arith.mulf %a, %b : f32
        %s = arith.addf %c, %p : f32
        affine.store %s, %Yg[%m, %n] : memref<64x256xf32>
      }
    }
  }
  return
}

// Both nests vectorize: the conv band carries a vector accumulator (ow stepped
// by 16) and the GEMM is broadcast-blocked (k iter_args of vector type).
// CHECK: affine.for %{{.*}} = 0 to 64 step 16
// CHECK: vector.broadcast
// CHECK: affine.vector_store %{{.*}}, %arg2
// The GEMM still vectorizes (broadcast kernel over its k reduction).
// CHECK: affine.vector_store %{{.*}}, %arg5
