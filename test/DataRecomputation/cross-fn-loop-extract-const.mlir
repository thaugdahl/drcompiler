// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// Strategy F.1 — constant index in reader. The reader's load uses a
// compile-time constant index, so the extraction should clone the
// constant into the caller and substitute it for the writer's IV.

module {
  // expected-remark @below {{cost-model:}}
  memref.global "private" @g : memref<8xi32> = uninitialized

  func.func private @writer(%base: i32) {
    %g = memref.get_global @g : memref<8xi32>
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c8 step %c1 {
      %ic = arith.index_cast %i : index to i32
      %v = arith.muli %base, %ic : i32
      memref.store %v, %g[%i] : memref<8xi32>
    }
    return
  }

  func.func private @reader() -> i32 {
    %g = memref.get_global @g : memref<8xi32>
    %c3 = arith.constant 3 : index
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[%c3] : memref<8xi32>
    return %v : i32
  }

  func.func @run(%base: i32) -> i32 {
    call @writer(%base) : (i32) -> ()
    // expected-remark @below {{interproc-cross: ACCEPT}}
    %r = call @reader() : () -> i32
    return %r : i32
  }
}

// CHECK-LABEL: func.func @run
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     scf.for
// CHECK:         call @writer
// The constant 3 is cloned into @run, then index_cast and muli mirror
// the writer's body for that single iteration.
// CHECK:         arith.constant 3 : index
// CHECK:         arith.index_cast
// CHECK:         arith.muli
// CHECK:         call @reader
