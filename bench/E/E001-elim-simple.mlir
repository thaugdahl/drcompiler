// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-buffer-elim=true dr-summary=true})' 2>&1 | FileCheck %s --check-prefix=SUM
// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-buffer-elim=true dr-erase-eliminated-buffers=true})' | FileCheck %s --check-prefix=ERASED

// E001: Single alloc, 1 store, 1 load, no escape → FEASIBLE + erasable.

// SUM: buffer-elim {{.*}}: FEASIBLE (escape=no-escape

// ERASED-LABEL: func.func @simple_elim
// ERASED-NOT:   memref.alloca
// ERASED-NOT:   memref.store
// ERASED:       %[[C:.+]] = arith.constant 42 : i32
// ERASED:       return %[[C]] : i32
func.func @simple_elim() -> i32 {
  %a = memref.alloca() : memref<i32>
  %c = arith.constant 42 : i32
  memref.store %c, %a[] : memref<i32>
  %v = memref.load %a[] : memref<i32>
  return %v : i32
}
