// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// External call between writer and reader may write any index of the
// escaping buffer. Const-index load must NOT classify SINGLE.

module {
  func.func private @opaque(memref<3xi64>)
  func.func @escape_then_load(%v: i64) -> i64 {
    %buf = memref.alloc() : memref<3xi64>
    %c0 = arith.constant 0 : index
    affine.store %v, %buf[%c0] : memref<3xi64>
    func.call @opaque(%buf) : (memref<3xi64>) -> ()
    // expected-remark @below {{load: LEAKED}}
    %r = affine.load %buf[%c0] : memref<3xi64>
    return %r : i64
  }
}
