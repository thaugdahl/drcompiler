// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// affine.for upper bound is a dynamic SSA value (not statically constant).
// IV must NOT fold to lb — even if the runtime trip is 1, we cannot prove
// it. Store coverage must be nullopt; the loop may run zero times, so the
// post-loop state gains an absent-path null sentinel and the load is
// LEAKED (would read uninit on the zero-trip path).

#id = affine_map<(d0) -> (d0)>

module {
  func.func @dynamic_trip(%n: index, %v: i64) -> i64 {
    %buf = memref.alloc() : memref<3xi64>
    %c0 = arith.constant 0 : index
    affine.for %i = 0 to %n {
      %ii = affine.apply #id(%i)
      affine.store %v, %buf[%ii] : memref<3xi64>
    }
    // expected-remark @below {{load: LEAKED}}
    %r = affine.load %buf[%c0] : memref<3xi64>
    return %r : i64
  }
}
