// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-footprint-analysis dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// C029: Footprint — operand memref is warm (recently accessed).
// Buffer = 1xi32, ALU = addi(1), 1 consumer.
// The stored value is computed from %src, and the intervening loop also
// reads %src, keeping it warm in cache.  Operand penalty = 0.
// Despite L1 eviction (32KB intervening), recompute is cheap.

module {
  func.func @footprint_operand_warm(%src: memref<8192xi32>) -> i32 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1 : i32

    // expected-remark @+1 {{cost-model: RECOMPUTE}}
    %buf = memref.alloc() : memref<1xi32>

    %x = memref.load %src[%c0] : memref<8192xi32>
    %val = arith.addi %x, %one : i32
    memref.store %val, %buf[%c0] : memref<1xi32>

    // Intervening loop reads from the SAME memref → keeps %src warm.
    %sum = arith.constant 0 : i32
    %r = affine.for %i = 0 to 8192 iter_args(%acc = %sum) -> (i32) {
      %v = memref.load %src[%i] : memref<8192xi32>
      %next = arith.addi %acc, %v : i32
      affine.yield %next : i32
    }

    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %a = memref.load %buf[%c0] : memref<1xi32>

    %result = arith.addi %a, %r : i32
    memref.dealloc %buf : memref<1xi32>
    return %result : i32
  }
}

// CHECK-LABEL: func.func @footprint_operand_warm
// CHECK-NOT:     memref.load %alloc
// CHECK:         return
