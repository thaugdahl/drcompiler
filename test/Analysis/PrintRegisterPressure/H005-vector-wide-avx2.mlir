// RUN: dr-opt %s --pass-pipeline='builtin.module(print-register-pressure{cpu-cost-model-file=%S/Inputs/avx2.json})' -verify-diagnostics

// 16xf32 = 512 bits; on AVX2 each value needs 2 vec regs.  At addf: two
// operands + defined result = 3 values * 2 regs each -> peak vec = 6.
// expected-remark@below {{register-pressure: vadd16 strategy=excess-hot peak=(gp=0,fp=0,vec=6,pred=0) excess=(gp=0,fp=0,vec=0,pred=0) spill_cycles=0}}
func.func @vadd16(%a: vector<16xf32>, %b: vector<16xf32>) -> vector<16xf32> {
  %c = arith.addf %a, %b : vector<16xf32>
  return %c : vector<16xf32>
}
