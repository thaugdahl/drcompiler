// RUN: dr-opt %s --pass-pipeline='builtin.module(print-register-pressure{cpu-cost-model-file=%S/Inputs/avx512.json})' -verify-diagnostics

// i1 result of cmpi classifies into the Pred class.  Peak pred = 1.
// expected-remark@below {{register-pressure: cmp strategy=excess-hot peak=(gp=2,fp=0,vec=0,pred=1) excess=(gp=0,fp=0,vec=0,pred=0) spill_cycles=0}}
func.func @cmp(%a: i32, %b: i32) -> i1 {
  %p = arith.cmpi slt, %a, %b : i32
  return %p : i1
}
