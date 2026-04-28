// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// Strategy F.2: scf.for with constant bounds, store inside the loop body.
//
// @writer fills @g[0..4] inside an scf.for. @reader loads @g[%i].
// Strategy F.2 should clone the loop into a scratch buffer at the caller
// and rewrite the reader's load to read from the scratch.

module {
  // expected-remark @below {{cost-model:}}
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

  func.func private @reader(%i: index) -> i32 {
    %g = memref.get_global @g : memref<4xi32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[%i] : memref<4xi32>
    return %v : i32
  }

  func.func @run(%base: i32, %i: index) -> i32 {
    call @writer(%base) : (i32) -> ()
    // expected-remark @below {{interproc-cross: ACCEPT}}
    %r = call @reader(%i) : (index) -> i32
    return %r : i32
  }
}

// CHECK-LABEL: func.func @run
// CHECK:         call @writer
// A scratch alloca + cloned loop is materialized at the caller.
// CHECK:         %[[SCRATCH:.*]] = memref.alloca() : memref<4xi32>
// CHECK:         scf.for
// CHECK:           memref.store {{.*}}, %[[SCRATCH]]
// CHECK:         call @reader{{.*}}(%{{.*}}, %[[SCRATCH]]
