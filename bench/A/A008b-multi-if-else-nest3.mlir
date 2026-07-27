// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// A008b: nested if/else 3 levels deep, each branch stores different value.
// Uses affine.if for nesting since scf.if can't nest easily with new conds.

module {
  func.func @multi_if_else_nest3() {
    %alloc = memref.alloc() : memref<i32>
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c4 = arith.constant 4 : i32

    // always-true: (0 >= 0)
    affine.if affine_set<() : (0 >= 0)>() {
      // always-true nested
      affine.if affine_set<() : (0 >= 0)>() {
        // always-true nested
        affine.if affine_set<() : (0 >= 0)>() {
          memref.store %c1, %alloc[] : memref<i32>
        } else {
          memref.store %c2, %alloc[] : memref<i32>
        }
      } else {
        memref.store %c3, %alloc[] : memref<i32>
      }
    } else {
      memref.store %c4, %alloc[] : memref<i32>
    }
    // expected-remark @below {{load: MULTI}}
    %v = memref.load %alloc[] : memref<i32>
    memref.dealloc %alloc : memref<i32>
    return
  }
}
