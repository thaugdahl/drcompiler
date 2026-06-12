// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block{mr=8 nr=16 vl=16}))' | FileCheck %s

// A de-promoted interior 3x3 conv (constant bounds) -- a MULTI-loop reduction
// band (ic/kh/kw) under the spatial loop ow, accumulating into Y in memory --
// is vectorized along ow: the weight load becomes a broadcast, the input load
// (stride-1 in ow, offset by kw) a contiguous vector load, and the accumulator
// a vector<16> carried through the whole band as iter_args.  This is the WP-O2
// direct-conv kernel; the GEMM stages cannot reach it (ow does not directly
// enclose a single reduction loop).

// CHECK-LABEL: func.func @conv3x3
func.func @conv3x3(%in: memref<64x58x66xf32>, %w: memref<64x64x3x3xf32>, %Y: memref<64x56x64xf32>) {
  affine.for %oc = 0 to 64 {
    affine.for %oh = 0 to 56 {
      affine.for %ow = 0 to 64 {
        affine.for %ic = 0 to 64 {
          affine.for %kh = 0 to 3 {
            affine.for %kw = 0 to 3 {
              %i = affine.load %in[%ic, %oh + %kh, %ow + %kw] : memref<64x58x66xf32>
              %ww = affine.load %w[%oc, %ic, %kh, %kw] : memref<64x64x3x3xf32>
              %c = affine.load %Y[%oc, %oh, %ow] : memref<64x56x64xf32>
              %p = arith.mulf %i, %ww : f32
              %s = arith.addf %c, %p : f32
              affine.store %s, %Y[%oc, %oh, %ow] : memref<64x56x64xf32>
            }
          }
        }
      }
    }
  }
  return
}

// ow is re-stepped by the vector width.
// CHECK: affine.for %{{.*}} = 0 to 64 step 16

// The accumulator slab is vector-loaded once before the band.
// CHECK: affine.vector_load %arg2[{{.*}}] : memref<64x56x64xf32>, vector<16xf32>

// The band is rebuilt carrying a vector<16> iter_arg (no in-band scalar store).
// CHECK: affine.for %{{.*}} = 0 to 64 iter_args({{.*}}) -> (vector<16xf32>)
// CHECK:   affine.for %{{.*}} = 0 to 3 iter_args({{.*}}) -> (vector<16xf32>)
// CHECK:     affine.for %{{.*}} = 0 to 3 iter_args({{.*}}) -> (vector<16xf32>)
// CHECK:       affine.vector_load %arg0[{{.*}}] : memref<64x58x66xf32>, vector<16xf32>
// CHECK:       vector.broadcast %{{.*}} : f32 to vector<16xf32>
// CHECK:       arith.mulf %{{.*}}, %{{.*}}{{.*}}: vector<16xf32>
// CHECK:       arith.addf %{{.*}}, %{{.*}}{{.*}}: vector<16xf32>

// The final vector accumulator is stored once after the band.
// CHECK: affine.vector_store %{{.*}}, %arg2[{{.*}}] : memref<64x56x64xf32>, vector<16xf32>
