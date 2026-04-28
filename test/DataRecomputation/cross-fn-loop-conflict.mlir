// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// Strategy F.2 — caller-side conflict rejection.
//
// Same shape as cross-fn-loop-scf, but @run calls @clobber between
// @writer and @reader. @clobber writes to the same global @g. The plan
// would otherwise fire, but the per-caller interference check rejects it.

module {
  memref.global "private" @g : memref<4xi32> = uninitialized

  func.func private @writer(%base: i32) {
    %g = memref.get_global @g : memref<4xi32>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %ic = arith.index_cast %i : index to i32
      %v = arith.addi %base, %ic : i32
      memref.store %v, %g[%i] : memref<4xi32>
    }
    return
  }

  func.func private @clobber() {
    %g = memref.get_global @g : memref<4xi32>
    %z = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    memref.store %z, %g[%c0] : memref<4xi32>
    return
  }

  func.func private @reader(%i: index) -> i32 {
    %g = memref.get_global @g : memref<4xi32>
    // expected-remark @below {{load: MULTI}}
    %v = memref.load %g[%i] : memref<4xi32>
    return %v : i32
  }

  func.func @run(%base: i32, %i: index) -> i32 {
    call @writer(%base) : (i32) -> ()
    call @clobber() : () -> ()
    %r = call @reader(%i) : (index) -> i32
    return %r : i32
  }
}

// CHECK-LABEL: func.func @run
// No scratch buffer — the conflict-check rejected the materialization.
// CHECK-NOT: memref.alloca
// CHECK:         call @writer
// CHECK:         call @clobber
// CHECK:         call @reader
