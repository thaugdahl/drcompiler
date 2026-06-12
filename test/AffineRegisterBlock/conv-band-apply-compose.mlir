// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block{mr=8 nr=16 vl=16}))' | FileCheck %s

// onnx-mlir precomputes every load index via affine.apply: the input load is
// `in[.., apply(kh,oh), apply(kw,ow)]`, so `ow` is not a direct operand of the
// load and the stride analysis cannot see it.  C1 (WP-O2 part 3b) composes the
// applies into the band's own load/store maps before the conv-band checks, so
// a stride-1 apply-ed conv vectorizes (the group apply on oc folds into the
// accumulator map too), while a stride-2 stem composes to `ow*2 + kw - 3` and
// correctly stays scalar instead of being mis-broadcast.

// CHECK-LABEL: func.func @conv3x3_apply
func.func @conv3x3_apply(%in: memref<1x64x58x66xf32>, %w: memref<64x64x3x3xf32>, %Y: memref<1x64x56x64xf32>) {
  affine.for %n = 0 to 1 {
    affine.for %g = 0 to 1 {
      affine.for %oc = 0 to 64 {
        %ocg = affine.apply affine_map<(d0, d1) -> (d0 * 64 + d1)>(%g, %oc)
        affine.for %oh = 0 to 56 {
          affine.for %ow = 0 to 64 {
            affine.for %ic = 0 to 64 {
              affine.for %kh = 0 to 3 {
                affine.for %kw = 0 to 3 {
                  %c = affine.load %Y[%n, %ocg, %oh, %ow] : memref<1x64x56x64xf32>
                  %a = affine.apply affine_map<(d0, d1) -> (d0 + d1 - 1)>(%kh, %oh)
                  %b = affine.apply affine_map<(d0, d1) -> (d0 + d1 - 1)>(%kw, %ow)
                  %i = affine.load %in[%n, %ic, %a, %b] : memref<1x64x58x66xf32>
                  %ww = affine.load %w[%ocg, %ic, %kh, %kw] : memref<64x64x3x3xf32>
                  %p = arith.mulf %i, %ww : f32
                  %s = arith.addf %c, %p : f32
                  affine.store %s, %Y[%n, %ocg, %oh, %ow] : memref<1x64x56x64xf32>
                }
              }
            }
          }
        }
      }
    }
  }
  return
}

// The composed input load is stride-1 in ow (last result `kw + ow - 1`), so
// the band vectorizes: ow re-steps by 16, the accumulator (group apply folded
// to `oc' = g*64 + oc` in the map) is carried as a vector iter_arg.
// CHECK: affine.for %{{.*}} = 0 to 64 step 16
// CHECK: affine.vector_load %arg2[{{.*}} * 64 + {{.*}}] : memref<1x64x56x64xf32>, vector<16xf32>
// CHECK: affine.for %{{.*}} = 0 to 64 iter_args({{.*}}) -> (vector<16xf32>)
// CHECK: affine.vector_load %arg0[{{.*}} - 1] : memref<1x64x58x66xf32>, vector<16xf32>
// CHECK: vector.broadcast
// CHECK: affine.vector_store

// CHECK-LABEL: func.func @conv_stride2_apply
func.func @conv_stride2_apply(%in: memref<1x64x58x131xf32>, %w: memref<64x64x3x3xf32>, %Y: memref<1x64x56x64xf32>) {
  affine.for %n = 0 to 1 {
    affine.for %oc = 0 to 64 {
      affine.for %oh = 0 to 56 {
        affine.for %ow = 0 to 64 {
          affine.for %ic = 0 to 64 {
            affine.for %kh = 0 to 3 {
              affine.for %kw = 0 to 3 {
                %c = affine.load %Y[%n, %oc, %oh, %ow] : memref<1x64x56x64xf32>
                %a = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%kh, %oh)
                %b = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%kw, %ow)
                %i = affine.load %in[%n, %ic, %a, %b] : memref<1x64x58x131xf32>
                %ww = affine.load %w[%oc, %ic, %kh, %kw] : memref<64x64x3x3xf32>
                %p = arith.mulf %i, %ww : f32
                %s = arith.addf %c, %p : f32
                affine.store %s, %Y[%n, %oc, %oh, %ow] : memref<1x64x56x64xf32>
              }
            }
          }
        }
      }
    }
  }
  return
}

// Composed input index is `ow*2 + kw`: NOT stride-1 in ow -> no vector ops.
// CHECK-NOT: vector
