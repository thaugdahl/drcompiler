// RUN: dr-opt %s --pass-pipeline='builtin.module(dr-par-bubbles{par-spmd-perband})' | FileCheck %s
// RUN: dr-opt %s --pass-pipeline='builtin.module(dr-par-bubbles{par-spmd-perband},func.func(convert-par-to-scf))' | FileCheck %s --check-prefix=SCF

// Conv-shaped band in the PROMOTED form (dr-scalar-reduction-promote): the inner
// reduction carries its accumulator as an affine.for iter_arg (register, not a
// memref round-trip), and a DEGENERATE batch axis (`%n = 0 to 1`) sits above the
// parallel `%oc` axis (shardIdx > 0).  par-spmd-perband must still materialize ONE
// par.forall over %oc by:
//   (a) de-affining the iter_arg reduction loops into scf.for WITH iter_args, and
//   (b) mapping the once-iterating outer %n to its lower bound (sound interchange).
// This is the resnet50 batch-1 conv shape; getting it to forall (not par.critical)
// is what lets the fast promoted reduction stay parallel.
// CHECK-LABEL: func.func @conv_promoted
// CHECK:         par.region {
// CHECK:           par.forall([0], [8], [1]) {
// CHECK:           ^bb0(%[[OC:.*]]: index):
// CHECK:             scf.for
// CHECK:               %[[R0:.*]] = scf.for {{.*}} iter_args(%[[A0:.*]] = {{.*}}) -> (f32) {
// CHECK:                 %[[R1:.*]] = scf.for {{.*}} iter_args(%[[A1:.*]] = %[[A0]]) -> (f32) {
// CHECK:                   memref.load
// CHECK:                   memref.load
// CHECK:                   arith.mulf
// CHECK:                   arith.addf
// CHECK:                   scf.yield
// CHECK:                 scf.yield %[[R1]]
// CHECK:               memref.store %[[R0]]
// CHECK:             par.yield
// CHECK:           par.yield
// CHECK:         }

// SCF-LABEL: func.func @conv_promoted
// SCF:         scf.parallel
// SCF:           scf.for {{.*}} iter_args
// SCF-NOT:       affine.for
func.func @conv_promoted(%out: memref<1x8x4xf32>, %in: memref<1x4x6xf32>, %wt: memref<8x4x3xf32>) {
  %z = arith.constant 0.0 : f32
  affine.for %n = 0 to 1 {
    affine.for %oc = 0 to 8 {
      affine.for %ow = 0 to 4 {
        %acc = affine.for %ic = 0 to 4 iter_args(%a = %z) -> (f32) {
          %r = affine.for %kw = 0 to 3 iter_args(%b = %a) -> (f32) {
            %i = affine.load %in[%n, %ic, %ow + %kw] : memref<1x4x6xf32>
            %w = affine.load %wt[%oc, %ic, %kw] : memref<8x4x3xf32>
            %p = arith.mulf %i, %w : f32
            %s = arith.addf %b, %p : f32
            affine.yield %s : f32
          }
          affine.yield %r : f32
        }
        affine.store %acc, %out[%n, %oc, %ow] : memref<1x8x4xf32>
      }
    }
  }
  return
}
