// RUN: dr-opt %s --pass-pipeline='builtin.module(print-register-pressure{cpu-cost-model-file=%S/Inputs/tiny-budget.json strategy=graph-color})' -verify-diagnostics

// Same 4-arg pressure program; under graph-color the Chaitin-Briggs heuristic
// is applied per class.  With budget=2 and 5 GP values total, 3 will be
// spilled.  Each spilled value contributes spill_store + (live_len * reload).
// The deterministic outcome depends on the heuristic; this test just pins it.
// expected-remark@below {{register-pressure: pressure4 strategy=graph-color peak=(gp=5,fp=0,vec=0,pred=0) excess=(gp=6,fp=0,vec=0,pred=0) spill_cycles=34}}
func.func @pressure4(%a: i32, %b: i32, %c: i32, %d: i32) -> i32 {
  %ab = arith.addi %a, %b : i32
  %cd = arith.addi %c, %d : i32
  %r = arith.addi %ab, %cd : i32
  return %r : i32
}
