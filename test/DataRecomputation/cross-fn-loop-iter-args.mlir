// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// P1: cross-function loop materialization now admits loops with iter_args
// (loop-carried state), which buildLoopPlan previously rejected outright.
//
// The writer's scf.for carries a running sum (%acc) and stores the scan to the
// global each iteration, so the STORED value depends on the iter_arg. Two
// soundness requirements are exercised:
//   1. The init operand (%c0) is a constant, hence reproducible at the caller,
//      so the whole loop is admitted and region-cloned verbatim.
//   2. Single-iteration extraction (Strategy F.1) is UNSOUND here (you cannot
//      extract one step of a reduction), so it must be suppressed and the full
//      loop materialized into a scratch buffer (Strategy F.2).

module {
  // expected-remark @below {{cost-model: RECOMPUTE}}
  memref.global "private" @g : memref<8xi32> = uninitialized

  func.func private @writer(%base: i32) {
    %g = memref.get_global @g : memref<8xi32>
    %c0 = arith.constant 0 : i32
    %lb = arith.constant 0 : index
    %ub = arith.constant 8 : index
    %st = arith.constant 1 : index
    %final = scf.for %i = %lb to %ub step %st iter_args(%acc = %c0) -> (i32) {
      %ic = arith.index_cast %i : index to i32
      %sum = arith.addi %acc, %ic : i32
      memref.store %sum, %g[%i] : memref<8xi32>
      scf.yield %sum : i32
    }
    return
  }

  func.func private @reader(%i: index) -> i32 {
    %g = memref.get_global @g : memref<8xi32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[%i] : memref<8xi32>
    return %v : i32
  }

  func.func @run(%base: i32, %i: index) -> i32 {
    call @writer(%base) : (i32) -> ()
    // expected-remark @below {{interproc-cross: ACCEPT_INPLACE}}
    %r = call @reader(%i) : (index) -> i32
    return %r : i32
  }
}

// Full-loop materialization (F.2): a scratch alloca is filled by a clone of the
// whole iter_args loop, and the reader is rewired to read the scratch buffer.
// CHECK-LABEL: func.func private @reader
// CHECK-SAME:    %{{.*}}: memref<8xi32>
// CHECK-LABEL: func.func @run
// CHECK:         call @writer
// CHECK:         memref.alloca
// CHECK:         scf.for {{.*}} iter_args
// CHECK:           arith.addi
// CHECK:           memref.store
// CHECK:           scf.yield
// CHECK:         call @reader
