// RUN: dr-opt %s --pass-pipeline='builtin.module(print-register-pressure)' -verify-diagnostics

// Mixed integer + float + vector classes.  Each class tracked independently.
// At the addf op: gp={n,i}=2, fp={a,b,c,sum}=4, vec={v0}=1.
// expected-remark@below {{register-pressure: mixed strategy=excess-hot peak=(gp=2,fp=4,vec=1,pred=0) excess=(gp=0,fp=0,vec=0,pred=0) spill_cycles=0}}
func.func @mixed(%n: i32, %i: i32, %a: f32, %b: f32, %c: f32, %v0: vector<4xf32>) -> f32 {
  %ab = arith.addf %a, %b : f32
  %sum = arith.addf %ab, %c : f32
  return %sum : f32
}
