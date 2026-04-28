// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// Strategy F.2: nested scf.for. Outer loop's body contains an inner
// scf.for that wraps the targeted store. Materialization should clone
// the entire outer loop (which contains the inner one) into the scratch.

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

// CHECK-LABEL: func.func @run
// CHECK:         call @writer
// CHECK:         %[[SCRATCH:.*]] = memref.alloca() : memref<2x3xi32>
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             memref.store {{.*}}, %[[SCRATCH]]
// CHECK:         call @reader{{.*}}, %[[SCRATCH]]
