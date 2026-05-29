// RUN: dr-opt %s --pass-pipeline='builtin.module(print-register-pressure{cpu-cost-model-file=%S/Inputs/tiny-budget.json strategy=excess-hot trip-count=1000})' -verify-diagnostics

// Identical to H007 but trip-count=1000 instead of 10.  excess(3) * 5 * 1000 = 15000.
// Verifies trip-count multiplies linearly under ExcessHot.
// expected-remark@below {{register-pressure: pressure4 strategy=excess-hot peak=(gp=5,fp=0,vec=0,pred=0) excess=(gp=6,fp=0,vec=0,pred=0) spill_cycles=15000}}
func.func @pressure4(%a: i32, %b: i32, %c: i32, %d: i32) -> i32 {
  %ab = arith.addi %a, %b : i32
  %cd = arith.addi %c, %d : i32
  %r = arith.addi %ab, %cd : i32
  return %r : i32
}
