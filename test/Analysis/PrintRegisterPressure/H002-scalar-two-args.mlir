// RUN: dr-opt %s --pass-pipeline='builtin.module(print-register-pressure)' -verify-diagnostics

// At the addi op: %a, %b live-in plus the defined result %c — peak gp = 3.
// expected-remark@below {{register-pressure: add2 strategy=excess-hot peak=(gp=3,fp=0,vec=0,pred=0) excess=(gp=0,fp=0,vec=0,pred=0) spill_cycles=0}}
func.func @add2(%a: i32, %b: i32) -> i32 {
  %c = arith.addi %a, %b : i32
  return %c : i32
}
