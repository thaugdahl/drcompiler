// RUN: dr-opt %s --pass-pipeline='builtin.module(print-register-pressure{cpu-cost-model-file=%S/Inputs/tiny-budget.json strategy=sum-excess})' -verify-diagnostics

// Same program as H007 under SumExcess.  Per-point excess: 3 + 2 + 1 + 0 = 6.
// spill_cycles = sum_excess(6) * spill_reload(5) = 30.
// expected-remark@below {{register-pressure: pressure4 strategy=sum-excess peak=(gp=5,fp=0,vec=0,pred=0) excess=(gp=6,fp=0,vec=0,pred=0) spill_cycles=30}}
func.func @pressure4(%a: i32, %b: i32, %c: i32, %d: i32) -> i32 {
  %ab = arith.addi %a, %b : i32
  %cd = arith.addi %c, %d : i32
  %r = arith.addi %ab, %cd : i32
  return %r : i32
}
