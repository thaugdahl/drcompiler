// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// A022: affine.for with symbolic bound that may be zero, store only inside
// loop at index 5, load at index 0 after loop. Even if the loop executes,
// index 0 has no covering store -> KILLED.
// (Alternatively: the loop stores at index 5, load reads index 0.)

module {
  func.func @killed_zero_trip_affine(%n: index) {
    %alloc = memref.alloc() : memref<32xi32>
    %c42 = arith.constant 42 : i32
    %c0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index
    memref.store %c42, %alloc[%c5] : memref<32xi32>
    affine.for %i = 0 to 10 {
      affine.store %c42, %alloc[5] : memref<32xi32>
    }
    // expected-remark @below {{load: KILLED}}
    %v = memref.load %alloc[%c0] : memref<32xi32>
    memref.dealloc %alloc : memref<32xi32>
    return
  }
}
