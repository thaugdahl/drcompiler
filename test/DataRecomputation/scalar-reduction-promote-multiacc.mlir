// WP-G2 safety net: dr-scalar-reduction-promote handles a band with N
// independent memory accumulators (the register-block mr-jam leftover when a
// band slips vectorization), not just one.  Each accumulator becomes its own
// iter_arg so the band degrades to registers instead of round-tripping through
// DRAM once per reduction step.
//
// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(dr-scalar-reduction-promote))' | FileCheck %s

// A k-reduction whose innermost body carries four same-shape memory
// accumulators C[0..3, j] (an mr=4 jam of C[i,j] += w[i,k]*x[k,j]).
func.func @jammed4(%w: memref<8x32xf32>, %x: memref<32x16xf32>, %c: memref<8x16xf32>) {
  affine.for %j = 0 to 16 {
    affine.for %k = 0 to 32 {
      %c0 = affine.load %c[0, %j] : memref<8x16xf32>
      %w0 = affine.load %w[0, %k] : memref<8x32xf32>
      %x0 = affine.load %x[%k, %j] : memref<32x16xf32>
      %m0 = arith.mulf %w0, %x0 : f32
      %s0 = arith.addf %c0, %m0 : f32
      affine.store %s0, %c[0, %j] : memref<8x16xf32>
      %c1 = affine.load %c[1, %j] : memref<8x16xf32>
      %w1 = affine.load %w[1, %k] : memref<8x32xf32>
      %x1 = affine.load %x[%k, %j] : memref<32x16xf32>
      %m1 = arith.mulf %w1, %x1 : f32
      %s1 = arith.addf %c1, %m1 : f32
      affine.store %s1, %c[1, %j] : memref<8x16xf32>
      %c2 = affine.load %c[2, %j] : memref<8x16xf32>
      %w2 = affine.load %w[2, %k] : memref<8x32xf32>
      %x2 = affine.load %x[%k, %j] : memref<32x16xf32>
      %m2 = arith.mulf %w2, %x2 : f32
      %s2 = arith.addf %c2, %m2 : f32
      affine.store %s2, %c[2, %j] : memref<8x16xf32>
      %c3 = affine.load %c[3, %j] : memref<8x16xf32>
      %w3 = affine.load %w[3, %k] : memref<8x32xf32>
      %x3 = affine.load %x[%k, %j] : memref<32x16xf32>
      %m3 = arith.mulf %w3, %x3 : f32
      %s3 = arith.addf %c3, %m3 : f32
      affine.store %s3, %c[3, %j] : memref<8x16xf32>
    }
  }
  return
}

// All four accumulators are loaded once before the k-loop, carried as four
// iter_args, and stored once after -- no affine.store to %c inside the loop.
// CHECK-LABEL: func @jammed4
// CHECK:         affine.for %{{.*}} = 0 to 16 {
// CHECK-COUNT-4: affine.load {{.*}} : memref<8x16xf32>
// CHECK:         %{{.*}}:4 = affine.for %{{.*}} = 0 to 32 iter_args({{.*}}) -> (f32, f32, f32, f32) {
// CHECK-NOT:       affine.store {{.*}} : memref<8x16xf32>
// CHECK:         affine.yield %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32, f32, f32, f32
// CHECK:         }
// CHECK-COUNT-4: affine.store {{.*}} : memref<8x16xf32>
