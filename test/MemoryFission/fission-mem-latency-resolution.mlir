// Fission's `memLatency` / `cacheLineSize` follow the full three-tier
// resolution contract (CLI > JSON > built-in default).  Both used to be
// function-local literals (200 / 64) in MemoryFissionPass::runOnOperation, then
// JSON-only; the CLI tier they were missing is what this test pins.
//
// The shape is the contention test's (3 consumers, 1 MiB buffer, totalWS 5 MiB,
// per-consumer source reuse 2 MiB, computeCost 40).  With l3 = 4 MiB the
// materialized buffer's reload is priced at MEM latency (totalWS 5 MiB > 4 MiB)
// while the source re-read still fits the effective LLC, so the always-on
// source-thrash clause stays out of the way and the decision is exactly
//   materialize = 40 + 1 + 3*memLatency   vs   recompute = 3*40
// i.e. it flips at memLatency ~= 26.  A cheap memory (20) makes materializing
// worth it; the default 200 does not.
//
// The JSON also sets a 128 B cache line (the Apple M-series geometry that
// motivated threading it): inert on THIS decision, which prices the reload from
// the buffer-size latency tier rather than the stride-aware footprint, but it
// must be settable here for the day the shared stride path is consulted.

// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission{test-diagnostics l3-size=4194304 mem-latency=20})' -o /dev/null 2>&1 | FileCheck %s --check-prefix=CLI-CHEAP
// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission{test-diagnostics l3-size=4194304})' -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission{test-diagnostics cpu-cost-model-file=%S/Inputs/low-mem-latency.json})' -o /dev/null 2>&1 | FileCheck %s --check-prefix=JSON
// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission{test-diagnostics cpu-cost-model-file=%S/Inputs/low-mem-latency.json mem-latency=200})' -o /dev/null 2>&1 | FileCheck %s --check-prefix=CLI-WINS

// CLI-CHEAP: memory-fission: FISSION
// DEFAULT: memory-fission: SKIP
// JSON: memory-fission: FISSION
// CLI-WINS: memory-fission: SKIP

module {
  func.func @big(%x: memref<131072xf64>, %o1: memref<131072xf64>,
                 %o2: memref<131072xf64>, %o3: memref<131072xf64>) {
    %c1 = arith.constant 1.0 : f64
    affine.for %i = 0 to 131072 {
      %xi = affine.load %x[%i] : memref<131072xf64>
      %sq = arith.mulf %xi, %xi : f64
      %s  = arith.addf %sq, %c1 : f64
      %r  = math.sqrt %s : f64
      %dd = arith.addf %xi, %c1 : f64
      %d  = arith.divf %r, %dd : f64
      affine.store %d, %o1[%i] : memref<131072xf64>
    }
    affine.for %i = 0 to 131072 {
      %xi = affine.load %x[%i] : memref<131072xf64>
      %sq = arith.mulf %xi, %xi : f64
      %s  = arith.addf %sq, %c1 : f64
      %r  = math.sqrt %s : f64
      %dd = arith.addf %xi, %c1 : f64
      %d  = arith.divf %r, %dd : f64
      affine.store %d, %o2[%i] : memref<131072xf64>
    }
    affine.for %i = 0 to 131072 {
      %xi = affine.load %x[%i] : memref<131072xf64>
      %sq = arith.mulf %xi, %xi : f64
      %s  = arith.addf %sq, %c1 : f64
      %r  = math.sqrt %s : f64
      %dd = arith.addf %xi, %c1 : f64
      %d  = arith.divf %r, %dd : f64
      affine.store %d, %o3[%i] : memref<131072xf64>
    }
    return
  }
}
