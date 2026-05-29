// RUN: dr-opt %s --pass-pipeline='builtin.module(print-register-pressure)' -verify-diagnostics

// Nested affine loops.  Inside inner body: %A, %B, %i, %j live (gp=4),
// loaded value + addf result (fp=2 at addf), constant (fp=1).
// Peak fp = 3 at the addf op (%v + %s + result).
// expected-remark@below {{register-pressure: matscale strategy=excess-hot peak=(gp=4,fp=3,vec=0,pred=0) excess=(gp=0,fp=0,vec=0,pred=0) spill_cycles=0}}
func.func @matscale(%A: memref<16x16xf32>, %B: memref<16x16xf32>) {
  %s = arith.constant 2.0 : f32
  affine.for %i = 0 to 16 {
    affine.for %j = 0 to 16 {
      %v = affine.load %A[%i, %j] : memref<16x16xf32>
      %r = arith.mulf %v, %s : f32
      affine.store %r, %B[%i, %j] : memref<16x16xf32>
    }
  }
  return
}
