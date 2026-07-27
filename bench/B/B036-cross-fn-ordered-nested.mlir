// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// B036: Cross-fn ordered — nested loops in callee writer.
// Writer has nested scf.for storing to 2D global; reader loads at known indices.

module {
  // expected-remark @below {{cost-model:}}
  memref.global "private" @g : memref<2x3xi32> = uninitialized

  func.func private @writer(%base: i32) {
    %g = memref.get_global @g : memref<2x3xi32>
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c2 step %c1 {
      scf.for %j = %c0 to %c3 step %c1 {
        %ic = arith.index_cast %i : index to i32
        %jc = arith.index_cast %j : index to i32
        %s = arith.addi %ic, %jc : i32
        %v = arith.addi %base, %s : i32
        memref.store %v, %g[%i, %j] : memref<2x3xi32>
      }
    }
    return
  }

  func.func private @reader(%i: index, %j: index) -> i32 {
    %g = memref.get_global @g : memref<2x3xi32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[%i, %j] : memref<2x3xi32>
    return %v : i32
  }

  func.func @run(%base: i32, %i: index, %j: index) -> i32 {
    call @writer(%base) : (i32) -> ()
    // expected-remark @below {{interproc-cross: ACCEPT}}
    %r = call @reader(%i, %j) : (index, index) -> i32
    return %r : i32
  }
}

// Nested-loop F.1: both store indices are IVs, both load indices are reader args.
// CHECK-LABEL: func.func @run
// CHECK-NOT:     scf.for
// CHECK:         call @writer
// CHECK:         arith.index_cast
// CHECK:         arith.index_cast
// CHECK:         arith.addi
// CHECK:         arith.addi
// CHECK:         call @reader
