// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-buffer-elim dr-test-diagnostics dr-arch-handler=x86-64-avx2})' -verify-diagnostics

// Regression test for the estimateBufferSizeBytes vector-element crash.
// Prior to the fix, memref<NxvectorMxfXX> in a DR cost-model path would
// trip an assertion inside getElementTypeBitWidth.  The pass should now
// run cleanly and emit the standard diagnostic stream.

module {
  func.func @vec_buf(%x: vector<4xf32>) -> vector<4xf32> {
    %c0 = arith.constant 0 : index
    %one = arith.constant dense<1.0> : vector<4xf32>

    // expected-remark @below {{cost-model: RECOMPUTE}}
    // expected-remark @below {{buffer-elim: FEASIBLE}}
    %buf = memref.alloc() : memref<1xvector<4xf32>>

    %v1 = arith.addf %x, %one : vector<4xf32>
    %v2 = arith.addf %v1, %one : vector<4xf32>
    memref.store %v2, %buf[%c0] : memref<1xvector<4xf32>>

    // expected-remark @below {{load: SINGLE}}
    // expected-remark @below {{direct-forward: ACCEPT}}
    %v = memref.load %buf[%c0] : memref<1xvector<4xf32>>

    memref.dealloc %buf : memref<1xvector<4xf32>>
    return %v : vector<4xf32>
  }
}
