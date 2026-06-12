// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block{mr=8 nr=16 vl=16}))' | FileCheck %s

// A PADDED conv clamps kw by ow (kw in [max(-ow+1,0), min(-ow+57,3))): the 16
// ow-lanes of one vector would each need a different kw trip count, so the
// band cannot vectorize as-is.  C2 (WP-O2 part 3b) splits ow into
// [left border | interior | right border] where the interior [1, 55) is
// exactly the region where both clamps are provably inactive; the interior's
// kw bounds become the constants [0, 3) and the borders keep the original
// clamped band verbatim (a pure index-set split -- checksum-identical).
// kh is clamped by OH (identical across ow-lanes) and is left alone.

// CHECK-LABEL: func.func @conv3x3_padded
func.func @conv3x3_padded(%in: memref<64x58x58xf32>, %w: memref<64x64x3x3xf32>, %Y: memref<64x56x56xf32>) {
  affine.for %oc = 0 to 64 {
    affine.for %oh = 0 to 56 {
      affine.for %ow = 0 to 56 {
        affine.for %ic = 0 to 64 {
          affine.for %kh = max affine_map<(d0) -> (-d0 + 1, 0)>(%oh) to min affine_map<(d0) -> (-d0 + 57, 3)>(%oh) {
            affine.for %kw = max affine_map<(d0) -> (-d0 + 1, 0)>(%ow) to min affine_map<(d0) -> (-d0 + 57, 3)>(%ow) {
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

// Left border [0, 1): original clamped kw bounds (scalar).
// CHECK: affine.for %{{.*}} = 0 to 1 {
// CHECK:   affine.for %{{.*}} = max #map{{.*}} to min #map
// CHECK:     affine.for %{{.*}} = max #map{{.*}} to min #map

// Interior [1, 55): kw bounds replaced by the constants [0, 3); kh keeps its
// oh-clamp.  (Still scalar here -- 54 is not VL-divisible; the vl-remainder
// peel is C3.)
// CHECK: affine.for %{{.*}} = 1 to 55 {
// CHECK:   affine.for %{{.*}} = 0 to 64 {
// CHECK:     affine.for %{{.*}} = max #map{{.*}} to min #map
// CHECK:       affine.for %{{.*}} = 0 to 3 {

// Right border [55, 56): original clamped kw bounds again.
// CHECK: affine.for %{{.*}} = 55 to 56 {
// CHECK:   affine.for %{{.*}} = max #map{{.*}} to min #map
// CHECK:     affine.for %{{.*}} = max #map{{.*}} to min #map

// A sub-VL spatial extent (14x14 at VL=16) must NOT be split: the interior
// would be below VL, so the whole band stays one scalar loop.
// CHECK-LABEL: func.func @conv3x3_14
func.func @conv3x3_14(%in: memref<256x16x16xf32>, %w: memref<256x256x3x3xf32>, %Y: memref<256x14x14xf32>) {
  affine.for %oc = 0 to 256 {
    affine.for %oh = 0 to 14 {
      affine.for %ow = 0 to 14 {
        affine.for %ic = 0 to 256 {
          affine.for %kh = max affine_map<(d0) -> (-d0 + 1, 0)>(%oh) to min affine_map<(d0) -> (-d0 + 15, 3)>(%oh) {
            affine.for %kw = max affine_map<(d0) -> (-d0 + 1, 0)>(%ow) to min affine_map<(d0) -> (-d0 + 15, 3)>(%ow) {
              %i = affine.load %in[%ic, %oh + %kh, %ow + %kw] : memref<256x16x16xf32>
              %ww = affine.load %w[%oc, %ic, %kh, %kw] : memref<256x256x3x3xf32>
              %c = affine.load %Y[%oc, %oh, %ow] : memref<256x14x14xf32>
              %p = arith.mulf %i, %ww : f32
              %s = arith.addf %c, %p : f32
              affine.store %s, %Y[%oc, %oh, %ow] : memref<256x14x14xf32>
            }
          }
        }
      }
    }
  }
  return
}

// CHECK: affine.for %{{.*}} = 0 to 14 {
// CHECK-NOT: affine.for %{{.*}} = 1 to 13
// CHECK-NOT: vector
