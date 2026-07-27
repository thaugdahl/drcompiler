// RUN: dr-opt %s --allow-unregistered-dialect --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-buffer-elim=true dr-summary=true})' 2>&1 | FileCheck %s --check-prefix=SUM
// RUN: dr-opt %s --allow-unregistered-dialect --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-buffer-elim=true dr-erase-eliminated-buffers=true})' | FileCheck %s --check-prefix=ERASED

// E002: 1 store, 4 loads — all SINGLE provenance → FEASIBLE.

// SUM: buffer-elim {{.*}}: FEASIBLE (escape=no-escape, loads=4

// ERASED-LABEL: func.func @multi_load
// ERASED-NOT:   memref.alloca
// ERASED-NOT:   memref.store
// ERASED:       %[[C:.+]] = arith.constant 99 : i32
// ERASED:       "use.consume"(%[[C]], %[[C]], %[[C]], %[[C]])
func.func @multi_load() {
  %a = memref.alloca() : memref<i32>
  %c = arith.constant 99 : i32
  memref.store %c, %a[] : memref<i32>
  %v0 = memref.load %a[] : memref<i32>
  %v1 = memref.load %a[] : memref<i32>
  %v2 = memref.load %a[] : memref<i32>
  %v3 = memref.load %a[] : memref<i32>
  "use.consume"(%v0, %v1, %v2, %v3) : (i32, i32, i32, i32) -> ()
  return
}
