// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block{mr=2 nr=2 vectorize=false cache-tile=true mc=64 nc=64 kc=64}))' | FileCheck %s

// A triangular *reduction* band (trmm shape): C[i][j] = sum_{k>=i} A[i][k]*B[k][j].
// The reduction k's LOWER bound depends on the outer spatial IV i (k = i..N), so
// the mr rows of an i-strip have different k-ranges and unroll-and-jam can't fuse
// them.  The pass distributes the C[i][j]=0 init sibling, then diagonal-peels the
// reduction: strip i by mr, build a MAIN part (k = ii+mr-1 .. N, uniform across
// the strip -> register-blockable) plus a scalar CORNER (k = ii+i' .. ii+mr-1,
// ragged).  This proves trmm register-blocks; the spike shows ~2.4x vs clang.

module {
  func.func @trmm(%A: memref<8x8xf64>, %B: memref<8x8xf64>, %C: memref<8x8xf64>) {
    %cst = arith.constant 0.0 : f64
    affine.for %i = 0 to 8 {
      affine.for %j = 0 to 8 {
        affine.store %cst, %C[%i, %j] : memref<8x8xf64>
      }
      affine.for %k = affine_map<(d0) -> (d0)>(%i) to 8 {
        affine.for %j = 0 to 8 {
          %a = affine.load %A[%i, %k] : memref<8x8xf64>
          %b = affine.load %B[%k, %j] : memref<8x8xf64>
          %c = affine.load %C[%i, %j] : memref<8x8xf64>
          %p = arith.mulf %a, %b : f64
          %s = arith.addf %c, %p : f64
          affine.store %s, %C[%i, %j] : memref<8x8xf64>
        }
      }
    }
    return
  }
}

// Strip-mined i (step mr=2).  The MAIN part register-blocks: the k-reduction loop
// (lower bound the strip-relative ii+mr-1) carries the mr*nr accumulators as
// iter_args.
// CHECK: affine.for %{{.*}} = 0 to 8 step 2
// CHECK: affine.for %{{.*}} = #{{.*}}(%{{.*}}) to 8 iter_args({{.*}}) -> (f64, f64, f64, f64)
// CHECK: affine.yield %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f64, f64, f64, f64
// The scalar CORNER keeps the ragged k range (a non-constant upper bound).
// CHECK: affine.for %{{.*}} = #{{.*}}(%{{.*}}) to #{{.*}}(%{{.*}}) {
