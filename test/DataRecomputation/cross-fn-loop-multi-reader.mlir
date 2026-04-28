// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// Strategy F.2: multiple loads in the same reader sharing one
// materialization. Two loads of @g (different indices) get rewritten to
// load from the same scratch buffer.

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

  func.func private @reader(%i: index, %j: index) -> i32 {
    %g = memref.get_global @g : memref<4xi32>
    // expected-remark @below {{load: SINGLE}}
    %a = memref.load %g[%i] : memref<4xi32>
    // expected-remark @below {{load: SINGLE}}
    %b = memref.load %g[%j] : memref<4xi32>
    %r = arith.addi %a, %b : i32
    return %r : i32
  }

  func.func @run(%base: i32, %i: index, %j: index) -> i32 {
    call @writer(%base) : (i32) -> ()
    // expected-remark @below {{interproc-cross: ACCEPT}}
    %r = call @reader(%i, %j) : (index, index) -> i32
    return %r : i32
  }
}

// CHECK-LABEL: func.func private @reader
// Both loads now use the new memref arg.
// CHECK:         memref.load %arg2
// CHECK:         memref.load %arg2

// CHECK-LABEL: func.func @run
// Single scratch alloca + single cloned loop is shared by both reader loads.
// CHECK:         call @writer
// CHECK:         %[[SCRATCH:.*]] = memref.alloca() : memref<4xi32>
// CHECK:         scf.for
// CHECK-NOT:     memref.alloca
// CHECK:         call @reader{{.*}}, %[[SCRATCH]]
