// RUN: dr-opt %s --allow-unregistered-dialect --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-cost-model=true dr-buffer-elim=true dr-summary=true})' 2>&1 | FileCheck %s --check-prefix=SUM

// E014: Small buffer (L1-resident), expensive ALU (division chain), 3 loads.
// Per-load cost model vetoes recomputation because recomputeCost > keepCost.
// Loads survive → buffer-elim INFEASIBLE (remaining loads still in IR).

// SUM: buffer-elim {{.*}}: INFEASIBLE (escape=no-escape, loads=3, remaining=3
func.func @expensive_remat(%x: i32) {
  %a = memref.alloca() : memref<i32>
  // Division costs 15 cycles; two divisions = 30 cycles ALU.
  %d1 = arith.divsi %x, %x : i32
  %d2 = arith.divsi %d1, %x : i32
  memref.store %d2, %a[] : memref<i32>
  %v0 = memref.load %a[] : memref<i32>
  %v1 = memref.load %a[] : memref<i32>
  %v2 = memref.load %a[] : memref<i32>
  "use.consume"(%v0, %v1, %v2) : (i32, i32, i32) -> ()
  return
}
