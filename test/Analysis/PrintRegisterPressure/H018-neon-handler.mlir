// RUN: dr-opt %s --pass-pipeline='builtin.module(print-register-pressure{cpu-cost-model-file=%S/Inputs/neon.json})' -verify-diagnostics

// vector<4xf32> = 128 bits, fits in a single NEON Q-reg (128-bit width).
// At addf: 2 operands + result -> peak vec = 3.
// expected-remark@below {{register-pressure: vadd_neon strategy=excess-hot peak=(gp=0,fp=0,vec=3,pred=0) excess=(gp=0,fp=0,vec=0,pred=0) spill_cycles=0}}
func.func @vadd_neon(%a: vector<4xf32>, %b: vector<4xf32>) -> vector<4xf32> {
  %c = arith.addf %a, %b : vector<4xf32>
  return %c : vector<4xf32>
}
