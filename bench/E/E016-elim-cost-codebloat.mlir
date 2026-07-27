// RUN: dr-opt %s --allow-unregistered-dialect --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-cost-model=true dr-buffer-elim=true dr-icache-soft-budget=4 dr-summary=true})' 2>&1 | FileCheck %s --check-prefix=SUM

// E016: 4 loads from a small buffer, stored value is a 6-division chain
// (90 cycles compute).  icache-soft-budget=4 adds a small code-bloat
// penalty; the new RegisterPressureAnalysis-driven path further inflates
// elimCost.  Verdict: INFEASIBLE.

// SUM: buffer-elim {{.*}}: INFEASIBLE (escape=no-escape{{.*}}keep=18, elim=102)
func.func @codebloat(%x: i32) {
  %a = memref.alloca() : memref<i32>
  %d0 = arith.divsi %x, %x : i32
  %d1 = arith.divsi %d0, %x : i32
  %d2 = arith.divsi %d1, %x : i32
  %d3 = arith.divsi %d2, %x : i32
  %d4 = arith.divsi %d3, %x : i32
  %d5 = arith.divsi %d4, %x : i32
  memref.store %d5, %a[] : memref<i32>
  %v0 = memref.load %a[] : memref<i32>
  %v1 = memref.load %a[] : memref<i32>
  %v2 = memref.load %a[] : memref<i32>
  %v3 = memref.load %a[] : memref<i32>
  "use.consume"(%v0, %v1, %v2, %v3) : (i32, i32, i32, i32) -> ()
  return
}
