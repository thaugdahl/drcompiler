// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// Strategy F.2: affine.for variant. Same shape as the scf.for test but
// using affine.for / affine.store / affine.load.

module {
  // expected-remark @below {{cost-model:}}
  memref.global "private" @g : memref<8xi32> = uninitialized

  func.func private @writer(%base: i32) {
    %g = memref.get_global @g : memref<8xi32>
    affine.for %i = 0 to 8 {
      %ic = arith.index_cast %i : index to i32
      %v = arith.addi %base, %ic : i32
      affine.store %v, %g[%i] : memref<8xi32>
    }
    return
  }

  func.func private @reader(%i: index) -> i32 {
    %g = memref.get_global @g : memref<8xi32>
    // expected-remark @below {{load: SINGLE}}
    %v = affine.load %g[%i] : memref<8xi32>
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
// F.1 fires (single-load reader, IV-only store index, reader-arg load
// index). No alloca, no loop clone.
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     affine.for
// CHECK:         call @writer
// CHECK:         arith.index_cast
// CHECK:         arith.addi
// CHECK:         call @reader
