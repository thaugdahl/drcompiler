// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-footprint-analysis dr-test-diagnostics})' -verify-diagnostics | FileCheck %s

// A2 (reuse-aware footprint cap). Between the producer store to %buf and its
// consumer loads, an intervening loop nest re-reads a small STATICALLY shaped
// array (%reuse, 256xf64 = 2KB) 1000*256 = 256000 times and writes %sink
// (also 2KB). The naive footprint = accesses * bytes = 1000*256*16 ≈ 4 MB,
// which would (wrongly) evict the buffer to DRAM (load latency 200) and force
// RECOMPUTE. But a loop cannot touch more DISTINCT bytes than the arrays it
// accesses contain: 2KB + 2KB + the 8-byte store to %buf = 4104 bytes, which
// stays in L1 (load latency 4). The cost model therefore correctly KEEPs.
//
// The pinned `storeToLoadFP=4104` locks the cap: reverting A2 yields
// storeToLoadFP≈4096008, load=200, and the decision flips to RECOMPUTE.
// (An identical program with %reuse/%sink as dynamic memref<?xf64> is
// uncappable and *does* report storeToLoadFP=4096008 / RECOMPUTE — the cap is
// what makes the static case resident.)

module {
  func.func @run(%x: f64) -> f64 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f64
    %eps = arith.constant 0.001 : f64

    // expected-remark @+1 {{cost-model: KEEP (compute=39, load=4, consumers=2, size=8, storeToLoadFP=4104, operandPenalty=0)}}
    %buf = memref.alloc() : memref<1xf64>
    %reuse = memref.alloc() : memref<256xf64>
    %sink = memref.alloc() : memref<256xf64>

    // Expensive producer (sqrt + div, critical-path cost 39).
    %sq = arith.mulf %x, %x : f64
    %sum = arith.addf %sq, %one : f64
    %root = math.sqrt %sum : f64
    %denom = arith.addf %x, %eps : f64
    %val = arith.divf %root, %denom : f64
    memref.store %val, %buf[%c0] : memref<1xf64>

    // Heavy temporal reuse of small static arrays — distinct bytes ≪ accesses.
    affine.for %k = 0 to 1000 {
      affine.for %j = 0 to 256 {
        %t = affine.load %reuse[%j] : memref<256xf64>
        affine.store %t, %sink[%j] : memref<256xf64>
      }
    }

    // expected-remark @+2 {{load: SINGLE}}
    // expected-remark @+1 {{cost-model: SKIP_LOAD (buffer kept)}}
    %a = memref.load %buf[%c0] : memref<1xf64>
    // expected-remark @+2 {{load: SINGLE}}
    // expected-remark @+1 {{cost-model: SKIP_LOAD (buffer kept)}}
    %b = memref.load %buf[%c0] : memref<1xf64>

    %r = arith.addf %a, %b : f64
    memref.dealloc %sink : memref<256xf64>
    memref.dealloc %reuse : memref<256xf64>
    memref.dealloc %buf : memref<1xf64>
    return %r : f64
  }
}

// The loads survive — the reuse loop stays in L1 once double-counting is removed.
// CHECK-LABEL: func.func @run
// CHECK:         math.sqrt
// CHECK:         memref.load
// CHECK:         memref.load
