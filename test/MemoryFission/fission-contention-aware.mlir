// Contention-aware fission decision (CONTENTION_AWARE_COSTMODEL.md).
//
// The SAME large-buffer candidate flips decision with the LLC-contention
// assumption: when isolated (llc-sharers=1) the materialized buffer would be
// LLC-resident and recompute re-reads the source cheaply, so fission is not
// worth the extra traffic -> SKIP.  Under contention (llc-sharers=20) the
// source's re-read reuse distance exceeds the *effective* (derated) LLC, so
// recompute would thrash DRAM N times and fission (reading the source once)
// wins -> FISSION.  Measured on a 7950X3D: this 1MB-buffer / 3-consumer shape
// regresses 0.72x isolated but speeds up 2.58x under heavy L3 contention.
//
// Buffer = 131072 * 8 = 1 MiB; working set ~5 MiB (>> 512 KiB guaranteed L2);
// per-consumer reuse distance ~2 MiB.  effective L3 = 32 MiB / sharers.

// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission{test-diagnostics llc-sharers=1})'  -o /dev/null 2>&1 | FileCheck %s --check-prefix=ISO
// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission{test-diagnostics llc-sharers=20})' -o /dev/null 2>&1 | FileCheck %s --check-prefix=CON

// ISO: memory-fission: SKIP
// CON: memory-fission: FISSION

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
