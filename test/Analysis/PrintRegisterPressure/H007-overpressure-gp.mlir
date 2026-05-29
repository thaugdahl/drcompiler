// RUN: dr-opt %s --pass-pipeline='builtin.module(print-register-pressure{cpu-cost-model-file=%S/Inputs/tiny-budget.json strategy=excess-hot trip-count=10})' -verify-diagnostics

// tiny-budget.json sets gp_budget=2, spill_reload=5.  At op1 (%ab = addi %a, %b):
// %a, %b, %c, %d live-in + defined %ab -> peak gp = 5; excess = 3.
// totalExcess is summed across all points: 3 + 2 + 1 + 0 = 6.
// ExcessHot uses the peak: excess(3) * spill_reload(5) * trip(10) = 150.
// expected-remark@below {{register-pressure: pressure4 strategy=excess-hot peak=(gp=5,fp=0,vec=0,pred=0) excess=(gp=6,fp=0,vec=0,pred=0) spill_cycles=150}}
func.func @pressure4(%a: i32, %b: i32, %c: i32, %d: i32) -> i32 {
  %ab = arith.addi %a, %b : i32
  %cd = arith.addi %c, %d : i32
  %r = arith.addi %ab, %cd : i32
  return %r : i32
}
