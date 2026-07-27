// RUN: dr-opt %s --allow-unregistered-dialect --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-buffer-elim=true dr-summary=true})' 2>&1 | FileCheck %s --check-prefix=SUM
// RUN: dr-opt %s --allow-unregistered-dialect --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-buffer-elim=true dr-erase-eliminated-buffers=true})' | FileCheck %s --check-prefix=KEPT

// E009: Alloc cast to index via memref.extract_aligned_pointer_as_index.
// The escape analysis does not whitelist this op → INFEASIBLE.

// SUM: buffer-elim {{.*}}: INFEASIBLE (escape=escapes

// KEPT-LABEL: func.func @ptr_cast
// KEPT:       memref.alloca
// KEPT:       memref.store
func.func @ptr_cast() {
  %a = memref.alloca() : memref<i32>
  %c = arith.constant 3 : i32
  memref.store %c, %a[] : memref<i32>
  %ptr = memref.extract_aligned_pointer_as_index %a : memref<i32> -> index
  "use.consume"(%ptr) : (index) -> ()
  return
}
