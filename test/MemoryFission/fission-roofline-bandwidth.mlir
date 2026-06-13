// Roofline / bandwidth term in the fission decision (CROSSCUTTING.md III.3, P1).
//
// The SAME candidate flips on the parallel BANDWIDTH assumption. A shared
// intermediate (compute cost 75) is recomputed by 3 consumers; materializing it
// into a buffer reloaded 3x is cheaper SINGLE-THREAD (materialize 196 <
// recompute 225) -> FISSION. But each reload streams the >L2 working set, and
// under a parallel workload that streaming competes for shared LLC/DRAM
// bandwidth: with 16 active threads at 16 B/cycle LLC bandwidth (1 B/cycle each)
// the per-reload cost is bandwidth-bound, not latency-bound, so materialize
// becomes far more expensive than recomputing per-thread -> SKIP.
//
// This is the cross-thread keep->recompute reversal the single-thread,
// latency-only model could not express. Without a `thread` block in the JSON the
// bandwidth term is disabled and the decision is byte-identical to before.

// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission{test-diagnostics})' \
// RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=SERIAL
// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission{test-diagnostics cpu-cost-model-file=%S/Inputs/threads-bw.json})' \
// RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=THREADED

// SERIAL:   memory-fission: FISSION (compute=75
// THREADED: memory-fission: SKIP (compute=75

module {
  func.func @rf(%x: memref<262144xf64>, %o1: memref<262144xf64>,
                %o2: memref<262144xf64>, %o3: memref<262144xf64>) {
    %c1 = arith.constant 1.0 : f64
    affine.for %i = 0 to 262144 {
      %xi = affine.load %x[%i] : memref<262144xf64>
      %sq = arith.mulf %xi, %xi : f64
      %s  = arith.addf %sq, %c1 : f64
      %r  = math.sqrt %s : f64
      %dd = arith.addf %xi, %c1 : f64
      %d  = arith.divf %r, %dd : f64
      %r2 = math.sqrt %d : f64
      %d2 = arith.divf %r2, %dd : f64
      affine.store %d2, %o1[%i] : memref<262144xf64>
    }
    affine.for %i = 0 to 262144 {
      %xi = affine.load %x[%i] : memref<262144xf64>
      %sq = arith.mulf %xi, %xi : f64
      %s  = arith.addf %sq, %c1 : f64
      %r  = math.sqrt %s : f64
      %dd = arith.addf %xi, %c1 : f64
      %d  = arith.divf %r, %dd : f64
      %r2 = math.sqrt %d : f64
      %d2 = arith.divf %r2, %dd : f64
      affine.store %d2, %o2[%i] : memref<262144xf64>
    }
    affine.for %i = 0 to 262144 {
      %xi = affine.load %x[%i] : memref<262144xf64>
      %sq = arith.mulf %xi, %xi : f64
      %s  = arith.addf %sq, %c1 : f64
      %r  = math.sqrt %s : f64
      %dd = arith.addf %xi, %c1 : f64
      %d  = arith.divf %r, %dd : f64
      %r2 = math.sqrt %d : f64
      %d2 = arith.divf %r2, %dd : f64
      affine.store %d2, %o3[%i] : memref<262144xf64>
    }
    return
  }
}
