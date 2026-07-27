// RUN: dr-opt %s --allow-unregistered-dialect --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-buffer-elim=true dr-summary=true})' 2>&1 | FileCheck %s --check-prefix=SUM

// E004: Caller allocates a buffer and passes it to a callee that stores.
// Caller then loads. Since the callee body is visible and only stores
// through the argument (no escape), the buffer should be FEASIBLE.

// SUM: buffer-elim {{.*}}: FEASIBLE (escape=no-escape

func.func @fill(%buf: memref<i32>) {
  %c = arith.constant 77 : i32
  memref.store %c, %buf[] : memref<i32>
  return
}

func.func @caller() -> i32 {
  %a = memref.alloca() : memref<i32>
  func.call @fill(%a) : (memref<i32>) -> ()
  %v = memref.load %a[] : memref<i32>
  "use.consume"(%v) : (i32) -> ()
  return %v : i32
}
