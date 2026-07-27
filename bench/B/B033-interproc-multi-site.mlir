// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// B033: Interproc multi-site — same callee called 3 times with different args.
// Each call site gets ACCEPT_SPECIALIZED.

module {
  // expected-remark @below {{cost-model:}}
  memref.global "private" @g : memref<i32> = uninitialized

  func.func private @writer(%x: i32) {
    %c10 = arith.constant 10 : i32
    %val = arith.addi %x, %c10 : i32
    %g = memref.get_global @g : memref<i32>
    memref.store %val, %g[] : memref<i32>
    return
  }

  func.func private @reader() -> i32 {
    %g = memref.get_global @g : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %g[] : memref<i32>
    return %v : i32
  }

  func.func @run(%a: i32, %b: i32, %c: i32) -> i32 {
    call @writer(%a) : (i32) -> ()
    // expected-remark @below {{interproc-cross: ACCEPT_SPECIALIZED}}
    %r1 = call @reader() : () -> i32

    call @writer(%b) : (i32) -> ()
    // expected-remark @below {{interproc-cross: ACCEPT_SPECIALIZED}}
    %r2 = call @reader() : () -> i32

    call @writer(%c) : (i32) -> ()
    // expected-remark @below {{interproc-cross: ACCEPT_SPECIALIZED}}
    %r3 = call @reader() : () -> i32

    %s = arith.addi %r1, %r2 : i32
    %r = arith.addi %s, %r3 : i32
    return %r : i32
  }
}

// CHECK-LABEL: func.func @run
// CHECK:         call @writer
// CHECK:         call @reader
// CHECK:         call @writer
// CHECK:         call @reader
// CHECK:         call @writer
// CHECK:         call @reader
