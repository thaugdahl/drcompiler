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

// Interior main [1, 49): kw bounds replaced by the constants [0, 3), kh keeps
// its oh-clamp, and the VL-divisible main is vectorized (C3 peel): the
// accumulator slab is carried as a vector<16> iter_arg through the whole band,
// the input load is contiguous in ow, the weight a broadcast.
// CHECK: affine.for %{{.*}} = 1 to 49 step 16 {
// CHECK:   affine.vector_load %arg2[{{.*}}] : memref<64x56x56xf32>, vector<16xf32>
// CHECK:   affine.for %{{.*}} = 0 to 64 iter_args({{.*}}) -> (vector<16xf32>)
// CHECK:     affine.for %{{.*}} = max #map{{.*}} to min #map{{.*}} iter_args({{.*}}) -> (vector<16xf32>)
// CHECK:       affine.for %{{.*}} = 0 to 3 iter_args({{.*}}) -> (vector<16xf32>)
// CHECK:         affine.vector_load %arg0[{{.*}}] : memref<64x58x58xf32>, vector<16xf32>
// CHECK:         vector.broadcast
// CHECK:   affine.vector_store

// Scalar vl-remainder tail [49, 55): constant kw bounds, no vector ops.
// CHECK: affine.for %{{.*}} = 49 to 55 {
// CHECK:   affine.for %{{.*}} = 0 to 64 {
// CHECK:     affine.for %{{.*}} = max #map{{.*}} to min #map
// CHECK:       affine.for %{{.*}} = 0 to 3 {

// Right border [55, 56): original clamped kw bounds again.
// CHECK: affine.for %{{.*}} = 55 to 56 {
// CHECK:   affine.for %{{.*}} = max #map{{.*}} to min #map
// CHECK:     affine.for %{{.*}} = max #map{{.*}} to min #map

// WP-G3 (per-band VL): a 14x14 interior (width 12) is below the pinned machine
// VL=16 but >= 8, so pickConvVL drops THIS band's VL to 8 and vectorizes the
// interior at VL=8 (before WP-G3 the band was left wholly scalar because the
// interior was sub-VL-16).  56/28 (interior >= 16) keep VL=16; an interior < 4
// still stays scalar.
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

// Interior [1, 13) split into a VL=8 main [1, 9) + scalar tail [9, 13), wrapped
// by the scalar left/right borders; the accumulator slab is a vector<8> iter_arg.
// CHECK: affine.for %{{.*}} = 0 to 1 {
// CHECK: affine.for %{{.*}} = 1 to 9 step 8 {
// CHECK:   affine.vector_load %{{.*}} : memref<256x14x14xf32>, vector<8xf32>
// CHECK:   affine.for %{{.*}} = 0 to 256 iter_args({{.*}}) -> (vector<8xf32>)
// CHECK:     affine.vector_load %{{.*}} : memref<256x16x16xf32>, vector<8xf32>
// CHECK:     vector.broadcast
