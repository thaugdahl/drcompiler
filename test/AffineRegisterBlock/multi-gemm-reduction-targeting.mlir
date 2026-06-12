// WP-G2: Stage 3 re-finds THIS band's reduction by its accumulator memref after
// the mr-jam, not the global-first reduction.  The old global
// findReductionLoopUnder(func) mispaired once any band was left un-vectorized
// (resnet50's residual blocks emit several same-shape 1024x196 / 256x196 GEMMs):
// the global walk returned a stranded earlier band, so the current band's jammed
// accumulators were never vectorized -- a cascade that left 16 of 33 GEMMs
// jammed-scalar accumulating in DRAM.  Targeting by the band's own accumulator
// alloc (unique per band, stable across the jam) fixes the pairing.
//
// DEFINITIVE evidence is the resnet50 end-to-end bench (17 -> 33 GEMMs
// vectorized; 1.207 -> 1.025 s, 1.069x -> 1.259x vs onnx-mlir --O3); this case
// is a forward guard that every GEMM in a multi-contraction function reaches the
// explicit broadcast vector kernel.
//
// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(affine-register-block{mr=8 nr=16}))' | FileCheck %s

// Four 1x1-conv GEMMs (reduce 1024->256, expand 256->1024) over two residual
// blocks at N=196, in the demoted memref-accumulator form register-block matches.
func.func @two_blocks(%n: index,
    %wr0: memref<256x1024xf32>, %xr0: memref<?x1024x196xf32>, %ar0: memref<?x256x196xf32>,
    %we0: memref<1024x256xf32>, %xe0: memref<?x256x196xf32>, %ae0: memref<?x1024x196xf32>,
    %wr1: memref<256x1024xf32>, %xr1: memref<?x1024x196xf32>, %ar1: memref<?x256x196xf32>,
    %we1: memref<1024x256xf32>, %xe1: memref<?x256x196xf32>, %ae1: memref<?x1024x196xf32>) {
  %z = arith.constant 0.0 : f32
  affine.for %b = 0 to %n { affine.for %o = 0 to 256 { affine.for %j = 0 to 196 {
    affine.store %z, %ar0[%b, %o, %j] : memref<?x256x196xf32> } } }
  affine.for %b = 0 to %n { affine.for %o = 0 to 256 { affine.for %j = 0 to 196 {
    affine.for %k = 0 to 1024 {
      %a = affine.load %ar0[%b, %o, %j] : memref<?x256x196xf32>
      %w = affine.load %wr0[%o, %k] : memref<256x1024xf32>
      %x = affine.load %xr0[%b, %k, %j] : memref<?x1024x196xf32>
      %m = arith.mulf %w, %x : f32
      %s = arith.addf %a, %m : f32
      affine.store %s, %ar0[%b, %o, %j] : memref<?x256x196xf32> } } } }
  affine.for %b = 0 to %n { affine.for %o = 0 to 1024 { affine.for %j = 0 to 196 {
    affine.store %z, %ae0[%b, %o, %j] : memref<?x1024x196xf32> } } }
  affine.for %b = 0 to %n { affine.for %o = 0 to 1024 { affine.for %j = 0 to 196 {
    affine.for %k = 0 to 256 {
      %a = affine.load %ae0[%b, %o, %j] : memref<?x1024x196xf32>
      %w = affine.load %we0[%o, %k] : memref<1024x256xf32>
      %x = affine.load %xe0[%b, %k, %j] : memref<?x256x196xf32>
      %m = arith.mulf %w, %x : f32
      %s = arith.addf %a, %m : f32
      affine.store %s, %ae0[%b, %o, %j] : memref<?x1024x196xf32> } } } }
  affine.for %b = 0 to %n { affine.for %o = 0 to 256 { affine.for %j = 0 to 196 {
    affine.store %z, %ar1[%b, %o, %j] : memref<?x256x196xf32> } } }
  affine.for %b = 0 to %n { affine.for %o = 0 to 256 { affine.for %j = 0 to 196 {
    affine.for %k = 0 to 1024 {
      %a = affine.load %ar1[%b, %o, %j] : memref<?x256x196xf32>
      %w = affine.load %wr1[%o, %k] : memref<256x1024xf32>
      %x = affine.load %xr1[%b, %k, %j] : memref<?x1024x196xf32>
      %m = arith.mulf %w, %x : f32
      %s = arith.addf %a, %m : f32
      affine.store %s, %ar1[%b, %o, %j] : memref<?x256x196xf32> } } } }
  affine.for %b = 0 to %n { affine.for %o = 0 to 1024 { affine.for %j = 0 to 196 {
    affine.store %z, %ae1[%b, %o, %j] : memref<?x1024x196xf32> } } }
  affine.for %b = 0 to %n { affine.for %o = 0 to 1024 { affine.for %j = 0 to 196 {
    affine.for %k = 0 to 256 {
      %a = affine.load %ae1[%b, %o, %j] : memref<?x1024x196xf32>
      %w = affine.load %we1[%o, %k] : memref<1024x256xf32>
      %x = affine.load %xe1[%b, %k, %j] : memref<?x256x196xf32>
      %m = arith.mulf %w, %x : f32
      %s = arith.addf %a, %m : f32
      affine.store %s, %ae1[%b, %o, %j] : memref<?x1024x196xf32> } } } }
  return
}

// All four reduction bands must reach the explicit vector micro-kernel: a
// vector iter_args reduction loop each (none left as a scalar memref
// accumulator round-tripping through DRAM per k-iteration).
// CHECK-COUNT-4: affine.for %{{.*}} iter_args({{.*}}vector<8xf32>
// CHECK-NOT: affine.for %{{.*}} iter_args({{.*}}vector<8xf32>
