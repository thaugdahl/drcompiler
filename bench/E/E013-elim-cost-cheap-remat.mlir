// RUN: dr-opt %s --allow-unregistered-dialect --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-cost-model=true dr-buffer-elim=true dr-summary=true})' 2>&1 | FileCheck %s --check-prefix=SUM
// RUN: dr-opt %s --allow-unregistered-dialect --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-cost-model=true dr-buffer-elim=true dr-erase-eliminated-buffers=true})' | FileCheck %s --check-prefix=ERASED

// E013: Large buffer (> L1), cheap computation (constant). The keepCost
// dominates because of L2 latency + alloc overhead → eliminate is cheaper.

// SUM: buffer-elim {{.*}}: FEASIBLE (escape=no-escape

// ERASED-LABEL: func.func @cheap_remat
// ERASED-NOT:   memref.alloc
// ERASED:       arith.constant 42 : i32
func.func @cheap_remat() {
  %a = memref.alloc() : memref<10000xi32>
  %c = arith.constant 42 : i32
  affine.store %c, %a[0] : memref<10000xi32>
  %v = affine.load %a[0] : memref<10000xi32>
  "use.consume"(%v) : (i32) -> ()
  memref.dealloc %a : memref<10000xi32>
  return
}
