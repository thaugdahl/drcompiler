// RUN: dr-opt %s --allow-unregistered-dialect --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-cost-model=true dr-buffer-elim=true dr-reg-budget-gp=4 dr-spill-reload=4 dr-summary=true})' 2>&1 | FileCheck %s --check-prefix=SUM

// E015: Remat tree has 10 addi ops, gp-budget=4.  Under the
// RegisterPressureAnalysis-driven elim path, the cloned chain pushes
// per-program-point live GP counts past the budget, inflating elimCost
// above keepCost.  Verdict: INFEASIBLE.

// SUM: buffer-elim {{.*}}: INFEASIBLE (escape=no-escape, loads=1, remaining=0{{.*}}keep=6, elim=94)
func.func @regpressure(%x: i32) {
  %a = memref.alloca() : memref<i32>
  %a0 = arith.addi %x, %x : i32
  %a1 = arith.addi %a0, %x : i32
  %a2 = arith.addi %a1, %x : i32
  %a3 = arith.addi %a2, %x : i32
  %a4 = arith.addi %a3, %x : i32
  %a5 = arith.addi %a4, %x : i32
  %a6 = arith.addi %a5, %x : i32
  %a7 = arith.addi %a6, %x : i32
  %a8 = arith.addi %a7, %x : i32
  %a9 = arith.addi %a8, %x : i32
  memref.store %a9, %a[] : memref<i32>
  %v = memref.load %a[] : memref<i32>
  "use.consume"(%v) : (i32) -> ()
  return
}
