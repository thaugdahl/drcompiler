// RUN: dr-opt %s --allow-unregistered-dialect --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-cost-model=true dr-buffer-elim=true dr-buffer-elim-drives-strategies=true dr-summary=true})' 2>&1 | FileCheck %s --check-prefix=SUM
// RUN: dr-opt %s --allow-unregistered-dialect --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-cost-model=true dr-buffer-elim=true dr-buffer-elim-drives-strategies=true dr-erase-eliminated-buffers=true})' | FileCheck %s --check-prefix=ERASED

// E020: Per-load cost model says KEEP (expensive ALU with 3 consumers on
// small L1 latency makes recomputeCost > keepCost per-load). But the
// buffer-elim rollup sees the alloc overhead (200 cycles for heap), store
// latency, and capacity penalty — making keepCost much larger. With
// dr-buffer-elim-drives-strategies, the rollup overrides the per-load veto.

// SUM: buffer-elim-override
// SUM: buffer-elim {{.*}}: FEASIBLE (escape=no-escape

// ERASED-LABEL: func.func @drives
// ERASED-NOT:   memref.alloc
func.func @drives(%x: i32) {
  // 40000 bytes > L1(32768) → L2 latency = 12 for loads.
  %a = memref.alloc() : memref<10000xi32>
  %d1 = arith.divsi %x, %x : i32
  %d2 = arith.divsi %d1, %x : i32
  affine.store %d2, %a[0] : memref<10000xi32>
  %v0 = affine.load %a[0] : memref<10000xi32>
  %v1 = affine.load %a[0] : memref<10000xi32>
  %v2 = affine.load %a[0] : memref<10000xi32>
  "use.consume"(%v0, %v1, %v2) : (i32, i32, i32) -> ()
  memref.dealloc %a : memref<10000xi32>
  return
}
