// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-buffer-elim=true dr-summary=true})' 2>&1 | FileCheck %s --check-prefix=SUM
// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-buffer-elim=true dr-erase-eliminated-buffers=true})' | FileCheck %s --check-prefix=ERASED

// Single alloca, single store, single load. SINGLE provenance, no escape →
// buffer is eliminable. With explicit erase, the alloc + store are gone.

// SUM: buffer-elim {{.*}}: FEASIBLE (escape=no-escape

// ERASED-LABEL: func.func @single_load
// ERASED-NOT:   memref.alloca
// ERASED-NOT:   memref.store
// ERASED:       %[[C:.+]] = arith.constant 42 : i32
// ERASED:       return %[[C]] : i32
func.func @single_load() -> i32 {
  %a = memref.alloca() : memref<i32>
  %c = arith.constant 42 : i32
  memref.store %c, %a[] : memref<i32>
  %v = memref.load %a[] : memref<i32>
  return %v : i32
}
