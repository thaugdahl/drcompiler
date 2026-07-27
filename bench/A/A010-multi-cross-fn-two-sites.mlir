// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// A010: same callee called from 2 sites with different values -> MULTI.
// Caller stores to global, then callee also stores to global at same index.

module {
  memref.global "private" @g : memref<32xi32> = uninitialized

  func.func private @writer(%idx: i32, %val: i32) {
    %g = memref.get_global @g : memref<32xi32>
    %idxt = arith.index_cast %idx : i32 to index
    memref.store %val, %g[%idxt] : memref<32xi32>
    return
  }

  func.func @test() -> i32 {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %idx = arith.constant 1 : index
    %g = memref.get_global @g : memref<32xi32>
    memref.store %c1, %g[%idx] : memref<32xi32>
    call @writer(%c1, %c2) : (i32, i32) -> ()
    // expected-remark @below {{load: MULTI}}
    %v = memref.load %g[%idx] : memref<32xi32>
    return %v : i32
  }
}
