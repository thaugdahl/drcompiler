// RUN: dr-opt %s --pass-pipeline='builtin.module(print-register-pressure)' -verify-diagnostics

// Single affine loop loading and storing scalars.  At the mulf op the scalar
// %s, the loaded %v, and the new product %r are all live -> peak fp = 3.
// expected-remark@below {{register-pressure: scaleloop strategy=excess-hot peak=(gp=3,fp=3,vec=0,pred=0) excess=(gp=0,fp=0,vec=0,pred=0) spill_cycles=0}}
func.func @scaleloop(%A: memref<128xf32>, %B: memref<128xf32>) {
  %s = arith.constant 2.0 : f32
  affine.for %i = 0 to 128 {
    %v = affine.load %A[%i] : memref<128xf32>
    %r = arith.mulf %v, %s : f32
    affine.store %r, %B[%i] : memref<128xf32>
  }
  return
}
