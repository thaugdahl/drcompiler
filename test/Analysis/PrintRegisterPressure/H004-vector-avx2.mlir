// RUN: dr-opt %s --pass-pipeline='builtin.module(print-register-pressure{cpu-cost-model-file=%S/Inputs/avx2.json})' -verify-diagnostics

// Two 8xf32 vectors (1 YMM each) + defined result at addf -> peak vec = 3.
// expected-remark@below {{register-pressure: vadd strategy=excess-hot peak=(gp=0,fp=0,vec=3,pred=0) excess=(gp=0,fp=0,vec=0,pred=0) spill_cycles=0}}
func.func @vadd(%a: vector<8xf32>, %b: vector<8xf32>) -> vector<8xf32> {
  %c = arith.addf %a, %b : vector<8xf32>
  return %c : vector<8xf32>
}
