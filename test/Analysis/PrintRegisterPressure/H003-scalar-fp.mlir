// RUN: dr-opt %s --pass-pipeline='builtin.module(print-register-pressure)' -verify-diagnostics

// At first addf: %a, %b, %c live-in + defined result %d -> peak fp = 4.
// expected-remark@below {{register-pressure: addf3 strategy=excess-hot peak=(gp=0,fp=4,vec=0,pred=0) excess=(gp=0,fp=0,vec=0,pred=0) spill_cycles=0}}
func.func @addf3(%a: f32, %b: f32, %c: f32) -> f32 {
  %d = arith.addf %a, %b : f32
  %e = arith.addf %d, %c : f32
  return %e : f32
}
