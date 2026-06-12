// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-scalar-reduction-promote))' | FileCheck %s

// Inverse of dr-scalar-reduction-demote: a perfect reduction band whose
// accumulator lives in memory (one load/add/store round-trip per innermost
// iteration) is rebuilt carrying the accumulator as a scalar iter_arg through
// every level -- one load before the band, one store after.  This cleans up
// the bands demote created but register-block did not take (sub-VL convs,
// stride-2 convs, scalar conv borders/tails).

// CHECK-LABEL: func.func @band3
func.func @band3(%in: memref<64x58x58xf32>, %w: memref<64x64x3x3xf32>, %Y: memref<64x56x56xf32>) {
  affine.for %oc = 0 to 64 {
    affine.for %oh = 0 to 56 {
      affine.for %ow = 0 to 56 {
        affine.for %ic = 0 to 64 {
          affine.for %kh = 0 to 3 {
            affine.for %kw = 0 to 3 {
              %i = affine.load %in[%ic, %oh + %kh, %ow + %kw] : memref<64x58x58xf32>
              %ww = affine.load %w[%oc, %ic, %kh, %kw] : memref<64x64x3x3xf32>
              %c = affine.load %Y[%oc, %oh, %ow] : memref<64x56x56xf32>
              %p = arith.mulf %i, %ww : f32
              %s = arith.addf %c, %p : f32
              affine.store %s, %Y[%oc, %oh, %ow] : memref<64x56x56xf32>
            }
          }
        }
      }
    }
  }
  return
}

// The accumulator is loaded ONCE per (oc,oh,ow), carried through ic/kh/kw as
// an iter_arg, stored once after the band.
// CHECK: affine.for %{{.*}} = 0 to 56 {
// CHECK: affine.for %{{.*}} = 0 to 56 {
// CHECK-NEXT: %[[INIT:.*]] = affine.load %arg2
// CHECK-NEXT: %[[R:.*]] = affine.for %{{.*}} = 0 to 64 iter_args(%{{.*}} = %[[INIT]]) -> (f32)
// CHECK:        affine.for %{{.*}} = 0 to 3 iter_args({{.*}}) -> (f32)
// CHECK:          affine.for %{{.*}} = 0 to 3 iter_args(%[[A:.*]] = %{{.*}}) -> (f32)
// CHECK:            %[[S:.*]] = arith.addf %[[A]], %{{.*}} : f32
// CHECK-NEXT:       affine.yield %[[S]]
// CHECK: affine.store %[[R]], %arg2
// CHECK-NOT: affine.load %arg2[{{.*}}] : memref<64x56x56xf32>

// A maxnumf accumulator (pooling-style) promotes too -- the rewrite is
// generic over the reduction op.
// CHECK-LABEL: func.func @maxpool
func.func @maxpool(%in: memref<64x112x112xf32>, %Y: memref<64x56x56xf32>) {
  affine.for %c = 0 to 64 {
    affine.for %oh = 0 to 56 {
      affine.for %ow = 0 to 56 {
        affine.for %kh = 0 to 2 {
          affine.for %kw = 0 to 2 {
            %i = affine.load %in[%c, %oh * 2 + %kh, %ow * 2 + %kw] : memref<64x112x112xf32>
            %a = affine.load %Y[%c, %oh, %ow] : memref<64x56x56xf32>
            %m = arith.maxnumf %a, %i : f32
            affine.store %m, %Y[%c, %oh, %ow] : memref<64x56x56xf32>
          }
        }
      }
    }
  }
  return
}
// CHECK: iter_args
// CHECK: arith.maxnumf

// A second store in the body (not the accumulator) must NOT promote: the
// rebuild would reorder it relative to the accumulator store.
// CHECK-LABEL: func.func @second_store
func.func @second_store(%A: memref<128xf32>, %Y: memref<16xf32>, %T: memref<16x8xf32>) {
  affine.for %i = 0 to 16 {
    affine.for %k = 0 to 8 {
      %v = affine.load %A[%i * 8 + %k] : memref<128xf32>
      %c = affine.load %Y[%i] : memref<16xf32>
      %s = arith.addf %c, %v : f32
      affine.store %s, %Y[%i] : memref<16xf32>
      affine.store %v, %T[%i, %k] : memref<16x8xf32>
    }
  }
  return
}
// CHECK-NOT: affine.for %{{.*}} = 0 to 8 iter_args

// An already-promoted (iter_args) reduction is left untouched.
// CHECK-LABEL: func.func @already_ssa
func.func @already_ssa(%A: memref<16x8xf32>, %Y: memref<16xf32>) {
  %z = arith.constant 0.0 : f32
  affine.for %i = 0 to 16 {
    %r = affine.for %k = 0 to 8 iter_args(%a = %z) -> (f32) {
      %v = affine.load %A[%i, %k] : memref<16x8xf32>
      %s = arith.addf %a, %v : f32
      affine.yield %s : f32
    }
    affine.store %r, %Y[%i] : memref<16xf32>
  }
  return
}
// CHECK: iter_args(%{{.*}} = %{{.*}}) -> (f32)
// CHECK-NOT: affine.load %arg1
