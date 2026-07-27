// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-buffer-elim=true dr-summary=true})' 2>&1 | FileCheck %s --check-prefix=SUM
// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-buffer-elim=true dr-erase-eliminated-buffers=true})' | FileCheck %s --check-prefix=KEPT

// E011: Caller allocs, passes to a callee that forwards the buffer to an
// external function. The interprocedural analysis sees an opaque write
// (provenance includes nullptr → leaked=1). Escape analysis also sees
// escapes-call. Both block elimination.

// SUM: buffer-elim {{.*}}: INFEASIBLE {{.*}}leaked=1

// KEPT-LABEL: func.func @leaked
// KEPT:       memref.alloca
func.func private @extern_writer(memref<i32>) -> ()

func.func @middle(%buf: memref<i32>) {
  func.call @extern_writer(%buf) : (memref<i32>) -> ()
  return
}

func.func @leaked() -> i32 {
  %a = memref.alloca() : memref<i32>
  %c = arith.constant 1 : i32
  memref.store %c, %a[] : memref<i32>
  func.call @middle(%a) : (memref<i32>) -> ()
  %v = memref.load %a[] : memref<i32>
  return %v : i32
}
