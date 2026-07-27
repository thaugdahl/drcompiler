// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// F016: Mixed affine.for outer, scf.for inner loop nest.
// SCF inner loop indices not tracked precisely → LEAKED.

module {
  func.func @nest_mixed(%arg0: i32) -> i32 {
    %buf = memref.alloc() : memref<32x32xi32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    affine.for %i = 0 to 32 {
      scf.for %j = %c0 to %c32 step %c1 {
        %ic = arith.index_cast %i : index to i32
        %jc = arith.index_cast %j : index to i32
        %v = arith.addi %ic, %jc : i32
        %w = arith.addi %v, %arg0 : i32
        memref.store %w, %buf[%i, %j] : memref<32x32xi32>
      }
    }
    // expected-remark @below {{load: LEAKED}}
    %out = memref.load %buf[%c0, %c0] : memref<32x32xi32>
    memref.dealloc %buf : memref<32x32xi32>
    return %out : i32
  }
}
