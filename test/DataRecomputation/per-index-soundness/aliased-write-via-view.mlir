// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// Write through a memref.reinterpret_cast that aliases the same backing
// buffer. collectBaseMemrefs unifies view→source so the store is
// attributed to the same allocRoot. Store coverage is nullopt
// (throughView), load coverage is concrete; the load includes the
// nullopt-coverage entry under the universal-overlap rule. Provenance =
// {storeOp}, classification SINGLE. Documents that view-aliased writes
// are tracked rather than dropped — a soundness *prerequisite* for the
// per-index code path.

module {
  func.func @aliased_via_view(%v: i64) -> i64 {
    %buf = memref.alloc() : memref<3xi64>
    %view = memref.reinterpret_cast %buf to offset:[0], sizes:[3], strides:[1]
              : memref<3xi64> to memref<3xi64>
    %c0 = arith.constant 0 : index
    affine.store %v, %view[%c0] : memref<3xi64>
    // expected-remark @below {{load: SINGLE}}
    %r = affine.load %buf[%c0] : memref<3xi64>
    return %r : i64
  }
}
