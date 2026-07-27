// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics

// G009: ALU cost near L1 load latency (4cy).
// Computation: addi chain of 4 ops => ALU=4, L1 buffer => load=4.
// keepCost = 4+1+1*4 = 9, recomputeCost = 1*4 = 4.
// Recompute wins since 4 < 9.

module {
  func.func @alu_tie(%x: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : i32

    // Small buffer, fits in L1.
    // expected-remark @below {{cost-model: RECOMPUTE}}
    %buf = memref.alloc() : memref<1xi32>

    // 4-op add chain: ALU = 4 cycles.
    %v1 = arith.addi %x, %c1 : i32
    %v2 = arith.addi %v1, %c1 : i32
    %v3 = arith.addi %v2, %c1 : i32
    %v4 = arith.addi %v3, %c1 : i32
    memref.store %v4, %buf[%c0] : memref<1xi32>

    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %out = memref.load %buf[%c0] : memref<1xi32>
    memref.dealloc %buf : memref<1xi32>
    return %out : i32
  }
}
