// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-scalar-reduction-demote))' | FileCheck %s

// A single-level f32 add-reduction stored directly (onnx-mlir's 1x1 conv / GEMM
// shape, batch trip-1 promoted to const 0) is fissioned into a zero-init nest
// and a memref-accumulating reduction band -- the form affine-register-block
// vectorizes.

// CHECK-LABEL: func.func @conv1x1
func.func @conv1x1(%W: memref<256x256xf32>, %X: memref<1x256x196xf32>, %Y: memref<1x256x196xf32>) {
  %cst0 = arith.constant 0.0 : f32
  affine.for %oc = 0 to 256 {
    affine.for %p = 0 to 196 {
      %r = affine.for %ic = 0 to 256 iter_args(%a = %cst0) -> (f32) {
        %w = affine.load %W[%oc, %ic] : memref<256x256xf32>
        %x = affine.load %X[0, %ic, %p] : memref<1x256x196xf32>
        %m = arith.mulf %w, %x : f32
        %s = arith.addf %a, %m : f32
        affine.yield %s : f32
      }
      affine.store %r, %Y[0, %oc, %p] : memref<1x256x196xf32>
    }
  }
  return
}

// The init nest comes first: store the seed at the accumulator subscript, no
// reduction loop inside.
// CHECK:      affine.for %[[OC0:.*]] = 0 to 256 {
// CHECK-NEXT:   affine.for %[[P0:.*]] = 0 to 196 {
// CHECK-NEXT:     affine.store %cst, %arg2[0, %[[OC0]], %[[P0]]]
// CHECK-NEXT:   }
// CHECK-NEXT: }

// The reduction band: NO iter_args; the accumulator is loaded and stored in the
// loop body.
// CHECK:      affine.for %[[OC1:.*]] = 0 to 256 {
// CHECK-NEXT:   affine.for %[[P1:.*]] = 0 to 196 {
// CHECK-NEXT:     affine.for %[[IC:.*]] = 0 to 256 {
// CHECK-NOT:        iter_args
// CHECK:            %[[C:.*]] = affine.load %arg2[0, %[[OC1]], %[[P1]]]
// CHECK:            arith.mulf
// CHECK:            %[[S:.*]] = arith.addf %[[C]], %{{.*}}
// CHECK:            affine.store %[[S]], %arg2[0, %[[OC1]], %[[P1]]]

// -----

// A maxpool/softmax-style reduction (maxnumf, not addf) is NOT demoted -- it
// stays as iter_args.  CHECK that the iter_args loop survives unchanged.

// CHECK-LABEL: func.func @maxpool
func.func @maxpool(%X: memref<64xf32>, %Y: memref<1xf32>) {
  %cstn = arith.constant -3.40282347E+38 : f32
  %r = affine.for %i = 0 to 64 iter_args(%a = %cstn) -> (f32) {
    %x = affine.load %X[%i] : memref<64xf32>
    %s = arith.maxnumf %a, %x : f32
    affine.yield %s : f32
  }
  affine.store %r, %Y[0] : memref<1xf32>
  return
}
// CHECK: affine.for %{{.*}} iter_args(%{{.*}}) -> (f32)
// CHECK:   arith.maxnumf

// -----

// A NESTED reduction band (k -> l threading one accumulator, the conv shape) is
// demoted: the whole band is rebuilt with NO iter_args, accumulating into C in
// memory, with a separate zero-init nest.

// CHECK-LABEL: func.func @nested
func.func @nested(%A: memref<8x8x8xf32>, %B: memref<8x8x8xf32>, %C: memref<8x8xf32>) {
  %cst0 = arith.constant 0.0 : f32
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 8 {
      %r = affine.for %k = 0 to 8 iter_args(%a = %cst0) -> (f32) {
        %inner = affine.for %l = 0 to 8 iter_args(%b = %a) -> (f32) {
          %x = affine.load %A[%i, %k, %l] : memref<8x8x8xf32>
          %y = affine.load %B[%j, %k, %l] : memref<8x8x8xf32>
          %m = arith.mulf %x, %y : f32
          %s = arith.addf %b, %m : f32
          affine.yield %s : f32
        }
        affine.yield %inner : f32
      }
      affine.store %r, %C[%i, %j] : memref<8x8xf32>
    }
  }
  return
}
// Init nest first (store seed, no reduction loop):
// CHECK:      affine.for %{{.*}} = 0 to 8 {
// CHECK-NEXT:   affine.for %{{.*}} = 0 to 8 {
// CHECK-NEXT:     affine.store %cst, %arg2[%{{.*}}, %{{.*}}]
// The rebuilt band: both k and l loops present, NO iter_args, memref accumulate.
// CHECK:      affine.for %[[I:.*]] = 0 to 8 {
// CHECK-NEXT:   affine.for %[[J:.*]] = 0 to 8 {
// CHECK-NEXT:     affine.for %{{.*}} = 0 to 8 {
// CHECK-NEXT:       affine.for %{{.*}} = 0 to 8 {
// CHECK-NOT:          iter_args
// CHECK:              affine.load %arg2[%[[I]], %[[J]]]
// CHECK:              affine.store %{{.*}}, %arg2[%[[I]], %[[J]]]
