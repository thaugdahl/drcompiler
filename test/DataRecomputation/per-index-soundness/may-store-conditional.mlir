// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// Conditional affine.store at constant index 0 is a may-write. The load
// must NOT classify SINGLE — it could read prior contents (uninit here).

module {
  func.func @may_store_const_idx(%cond: i1, %v: i64) -> i64 {
    %buf = memref.alloc() : memref<3xi64>
    %c0 = arith.constant 0 : index
    scf.if %cond {
      affine.store %v, %buf[%c0] : memref<3xi64>
    }
    // expected-remark @below {{load: LEAKED}}
    %r = affine.load %buf[%c0] : memref<3xi64>
    return %r : i64
  }
}
