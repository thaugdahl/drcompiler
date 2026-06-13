// Roofline / bandwidth term in the DataRecomputation keep-vs-recompute cost
// (CROSSCUTTING.md III.3/III.5, P1). A KEPT buffer is reloaded by its consumers;
// when it spills the private L2 each reload streams from the shared LLC/DRAM,
// which under a parallel workload is bandwidth-bound. The per-buffer load cost
// therefore rises to the per-thread bandwidth floor, making "keep the buffer"
// lose to per-thread recompute (pure ALU) under contention -- the same
// streamCycles() mechanism the fission roofline test exercises end-to-end.
//
// The 2 MiB buffer (262144 x f64) spills L2, so the keep-side load cost reflects
// the bandwidth floor, and it differs by DEPLOYMENT MODE (III.4a):
//   SERIAL       no thread JSON  -> 40 cycles (L3 latency tier, byte-identical)
//   INTERSPERSED shared BW/16    -> 2 MiB / (16 B/cyc / 16) = capped 1e6
//   EXCLUSIVE    owns full BW    -> 2 MiB / 16 B/cyc = 131072 (16x cheaper)
// i.e. the same buffer is far cheaper to keep when the workload owns the machine
// (exclusive) than when it competes with co-tenants (interspersed).

// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-cost-model dr-recompute dr-summary})' \
// RUN:   2>&1 | FileCheck %s --check-prefix=SERIAL
// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-cost-model dr-recompute dr-summary cpu-cost-model-file=%S/../MemoryFission/Inputs/threads-bw.json})' \
// RUN:   2>&1 | FileCheck %s --check-prefix=THREADED
// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-cost-model dr-recompute dr-summary cpu-cost-model-file=%S/../MemoryFission/Inputs/threads-bw-exclusive.json})' \
// RUN:   2>&1 | FileCheck %s --check-prefix=EXCLUSIVE

// SERIAL:    DRSUM: buffer {{.*}}compute=100, load=40,
// THREADED:  DRSUM: buffer {{.*}}compute=100, load=1000000,
// EXCLUSIVE: DRSUM: buffer {{.*}}compute=100, load=131072,

func.func @dr(%x: memref<262144xf64>, %out: memref<8xf64>) {
  %buf = memref.alloc() : memref<262144xf64>
  affine.for %i = 0 to 262144 {
    %xi = affine.load %x[%i] : memref<262144xf64>
    %a = math.sqrt %xi : f64
    %b = math.sqrt %a : f64
    %c = math.sqrt %b : f64
    %d = math.sqrt %c : f64
    %e = math.sqrt %d : f64
    affine.store %e, %buf[%i] : memref<262144xf64>
  }
  affine.for %j = 0 to 8 {
    %v = affine.load %buf[%j] : memref<262144xf64>
    affine.store %v, %out[%j] : memref<8xf64>
  }
  return
}
