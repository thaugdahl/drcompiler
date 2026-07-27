// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// A005b: chained cross-function stores via globals (depth 3).
// writer_a -> writer_b -> writer_c, reader loads final result.
// Each callee's load is SINGLE within its own analysis context.

module {
  memref.global "private" @a : memref<i32> = uninitialized
  memref.global "private" @b : memref<i32> = uninitialized
  memref.global "private" @c : memref<i32> = uninitialized

  func.func private @writer_a(%va: i32) {
    %a = memref.get_global @a : memref<i32>
    memref.store %va, %a[] : memref<i32>
    return
  }

  func.func private @writer_b() {
    %a = memref.get_global @a : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %av = memref.load %a[] : memref<i32>
    %one = arith.constant 1 : i32
    %vb = arith.addi %av, %one : i32
    %b = memref.get_global @b : memref<i32>
    memref.store %vb, %b[] : memref<i32>
    return
  }

  func.func private @writer_c() {
    %b = memref.get_global @b : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %bv = memref.load %b[] : memref<i32>
    %two = arith.constant 2 : i32
    %vc = arith.muli %bv, %two : i32
    %c = memref.get_global @c : memref<i32>
    memref.store %vc, %c[] : memref<i32>
    return
  }

  func.func private @reader() -> i32 {
    %c = memref.get_global @c : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %c[] : memref<i32>
    return %v : i32
  }

  func.func @test(%va: i32) -> i32 {
    call @writer_a(%va) : (i32) -> ()
    call @writer_b() : () -> ()
    call @writer_c() : () -> ()
    %r = call @reader() : () -> i32
    return %r : i32
  }
}
