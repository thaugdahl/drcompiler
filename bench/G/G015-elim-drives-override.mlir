// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-cost-model dr-buffer-elim=true dr-summary=true})' 2>&1 | FileCheck %s --check-prefix=SUM
// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-cost-model dr-buffer-elim=true dr-buffer-elim-drives-strategies=true dr-erase-eliminated-buffers=true})' | FileCheck %s --check-prefix=DRIVEN

// G015: Per-load cost model says KEEP (many consumers with moderate ALU).
// But whole-buffer elim rollup with shared-subexpr discount says the buffer
// is cheaper to eliminate. With dr-buffer-elim-drives-strategies, the
// rollup overrides per-load decisions.

func.func @elim_override() -> i32 {
  %a = memref.alloca() : memref<3xi32>
  %c = arith.constant 42 : i32

  // 3 identical stores with same constant (distinctTrees=1).
  affine.store %c, %a[0] : memref<3xi32>
  affine.store %c, %a[1] : memref<3xi32>
  affine.store %c, %a[2] : memref<3xi32>

  // 3 loads. Cost model sees 3 consumers; per-element
  // constant so elim cost with shared subexpr is ~0.
  %v0 = affine.load %a[0] : memref<3xi32>
  %v1 = affine.load %a[1] : memref<3xi32>
  %v2 = affine.load %a[2] : memref<3xi32>

  %s0 = arith.addi %v0, %v1 : i32
  %s1 = arith.addi %s0, %v2 : i32
  return %s1 : i32
}

// Buffer elim reports FEASIBLE (cheap shared subexpr).
// SUM: buffer-elim {{.*}}: FEASIBLE

// With drives enabled, buffer is erased.
// DRIVEN-LABEL: func.func @elim_override
// DRIVEN-NOT:   memref.alloca
// DRIVEN-NOT:   affine.store
// DRIVEN:       arith.constant 42
// DRIVEN:       return
