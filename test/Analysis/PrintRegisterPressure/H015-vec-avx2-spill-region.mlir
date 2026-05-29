// RUN: dr-opt %s --pass-pipeline='builtin.module(print-register-pressure{cpu-cost-model-file=%S/Inputs/tiny-vec.json strategy=excess-hot trip-count=100})' -verify-diagnostics

// tiny-vec.json: vec_budget=2, spill_reload=5.  Three vec<8xf32> values live
// simultaneously at the addf: 3-2 = 1 excess.  cycles = 1 * 5 * 100 = 500.
// expected-remark@below {{register-pressure: tightvec strategy=excess-hot peak=(gp=0,fp=0,vec=3,pred=0) excess=(gp=0,fp=0,vec=1,pred=0) spill_cycles=500}}
func.func @tightvec(%a: vector<8xf32>, %b: vector<8xf32>) -> vector<8xf32> {
  %c = arith.addf %a, %b : vector<8xf32>
  return %c : vector<8xf32>
}
