// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block{mr=8 nr=16 vl=16}))' | FileCheck %s

// WP4 (COSTMODEL_V4_SPEC §5): the PolyBench symm j-body holds a scatter into
// C[k][j] (k<i) intermixed with a rank-0 temp2 reduction, then an epilogue
// writing C[i][j].  raiseSymmScatter splits it into two nests:
//   (A) the original nest with the scatter removed -- temp2 stays a scalar
//       reduction and the epilogue is preserved; and
//   (B) a fresh scatter nest already interchanged so the accumulation loop i is
//       INNERMOST (lower bound k+1), which the existing broadcast vectorizer
//       register-blocks into an mr x vl micro-kernel.
// Result is bit-identical (validated at XL: symm 1.75x, SINK matches baseline).

// CHECK-LABEL: func.func @symm

// Branch A: temp2 stays a rank-0 scalar reduction; no C[k][j] vector store here.
// CHECK:      affine.for %{{.*}} = 0 to 256 {
// CHECK-NEXT:   affine.for %{{.*}} = 0 to 256 {
// CHECK:          affine.store %{{.*}}, %{{.*}}[] : memref<f64>
// CHECK:          affine.for %{{.*}} = 0 to #map(%{{.*}}) {

// Branch B: interchanged scatter (i innermost, lb = k+1), register-blocked.
// CHECK:      affine.for %{{.*}} = 0 to 256 {
// CHECK-NEXT:   affine.for %{{.*}} = 0 to 256 step 16 {
// CHECK:          affine.vector_load %{{.*}} : memref<256x256xf64>, vector<16xf64>
// CHECK:          affine.for %{{.*}} = #map1(%{{.*}}) to 256 iter_args({{.*}}) -> (vector<16xf64>) {
// CHECK:            vector.broadcast %{{.*}} : f64 to vector<16xf64>
// CHECK:          affine.vector_store %{{.*}} : memref<256x256xf64>, vector<16xf64>

module {
  func.func @symm(%alpha: f64, %A: memref<256x256xf64>, %B: memref<256x256xf64>, %C: memref<256x256xf64>) {
    %t = memref.alloca() : memref<f64>
    %z = arith.constant 0.0 : f64
    affine.for %i = 0 to 256 {
      affine.for %j = 0 to 256 {
        affine.store %z, %t[] : memref<f64>
        affine.for %k = 0 to affine_map<(d0) -> (d0)>(%i) {
          %b = affine.load %B[%i, %j] : memref<256x256xf64>
          %ab = arith.mulf %alpha, %b : f64
          %a = affine.load %A[%i, %k] : memref<256x256xf64>
          %p = arith.mulf %ab, %a : f64
          %c = affine.load %C[%k, %j] : memref<256x256xf64>
          %s = arith.addf %c, %p : f64
          affine.store %s, %C[%k, %j] : memref<256x256xf64>
          %bk = affine.load %B[%k, %j] : memref<256x256xf64>
          %ak = affine.load %A[%i, %k] : memref<256x256xf64>
          %pk = arith.mulf %bk, %ak : f64
          %tl = affine.load %t[] : memref<f64>
          %ts = arith.addf %tl, %pk : f64
          affine.store %ts, %t[] : memref<f64>
        }
        %ce = affine.load %C[%i, %j] : memref<256x256xf64>
        %te = affine.load %t[] : memref<f64>
        affine.store %te, %C[%i, %j] : memref<256x256xf64>
      }
    }
    return
  }
}
