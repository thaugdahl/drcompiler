// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// A011: conditional stores through different subviews that both alias base[0].
// if-branch stores via sv1[0] (base offset 0), else-branch stores via sv2[0]
// (also base offset 0). Conditional -> MULTI provenance.

module {
  func.func @multi_view_alias(%cond: i1) {
    %alloc = memref.alloc() : memref<32xi32>
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c0 = arith.constant 0 : index

    %sv1 = memref.subview %alloc[0][16][1]
        : memref<32xi32> to memref<16xi32>
    %sv2 = memref.subview %alloc[0][16][1]
        : memref<32xi32> to memref<16xi32>

    scf.if %cond {
      memref.store %c1, %sv1[%c0] : memref<16xi32>
    } else {
      memref.store %c2, %sv2[%c0] : memref<16xi32>
    }
    // expected-remark @below {{load: MULTI}}
    %v = memref.load %alloc[%c0] : memref<32xi32>
    memref.dealloc %alloc : memref<32xi32>
    return
  }
}
