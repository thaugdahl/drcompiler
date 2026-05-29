// RUN: dr-opt %s --pass-pipeline='builtin.module(print-register-pressure{cpu-cost-model-file=%S/Inputs/avx512.json})' -verify-diagnostics

// Affine loop over wide vectors on AVX-512 (512-bit).  Each vector<16xf32>
// fits in a single ZMM.  Body: 2 vec values live at mulf peak.
// expected-remark@below {{register-pressure: vecscale strategy=excess-hot peak=(gp=3,fp=0,vec=3,pred=0) excess=(gp=0,fp=0,vec=0,pred=0) spill_cycles=0}}
func.func @vecscale(%A: memref<32xvector<16xf32>>, %B: memref<32xvector<16xf32>>, %s: vector<16xf32>) {
  affine.for %i = 0 to 32 {
    %v = affine.load %A[%i] : memref<32xvector<16xf32>>
    %r = arith.mulf %v, %s : vector<16xf32>
    affine.store %r, %B[%i] : memref<32xvector<16xf32>>
  }
  return
}
