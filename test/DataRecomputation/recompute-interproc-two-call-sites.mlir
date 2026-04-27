// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-test-diagnostics})' -verify-diagnostics

// S1-B: A single storeOp inside @write reaches two distinct call sites
// that both dominate the load.  interproceduralOrigins maps the store to
// two origins; without the stopgap, the remat loop picks whichever was
// analyzed first and may rematerialize using the wrong call-site
// operands -- substituting %c1 for a load that is actually fed by the
// second call-site (%c2).
//
// With the stopgap, the pass refuses to commit to either origin and
// emits SKIP_AMBIGUOUS_ORIGIN, leaving the load intact.

module {
  func.func private @write(%x: i32, %out: memref<i32>) {
    memref.store %x, %out[] : memref<i32>
    return
  }

  func.func @exploit() -> i32 {
    %a = memref.alloc() : memref<i32>
    %b = memref.alloc() : memref<i32>
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    call @write(%c1, %a) : (i32, memref<i32>) -> ()
    call @write(%c2, %b) : (i32, memref<i32>) -> ()
    // expected-remark @+2 {{load: SINGLE}}
    // expected-remark @+1 {{interproc: SKIP_AMBIGUOUS_ORIGIN}}
    %v = memref.load %b[] : memref<i32>
    memref.dealloc %a : memref<i32>
    memref.dealloc %b : memref<i32>
    return %v : i32
  }
}
