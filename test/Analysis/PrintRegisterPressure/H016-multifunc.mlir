// RUN: dr-opt %s --pass-pipeline='builtin.module(print-register-pressure)' -verify-diagnostics

// Multiple functions in a module; each gets its own remark.
// expected-remark@below {{register-pressure: small strategy=excess-hot peak=(gp=3,fp=0,vec=0,pred=0) excess=(gp=0,fp=0,vec=0,pred=0) spill_cycles=0}}
func.func @small(%a: i32, %b: i32) -> i32 {
  %c = arith.addi %a, %b : i32
  return %c : i32
}

// expected-remark@below {{register-pressure: big strategy=excess-hot peak=(gp=0,fp=4,vec=0,pred=0) excess=(gp=0,fp=0,vec=0,pred=0) spill_cycles=0}}
func.func @big(%a: f32, %b: f32, %c: f32) -> f32 {
  %d = arith.addf %a, %b : f32
  %e = arith.addf %d, %c : f32
  return %e : f32
}
