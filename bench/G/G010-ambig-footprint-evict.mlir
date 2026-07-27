// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s --check-prefix=NOFP
// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-footprint-analysis dr-test-diagnostics})' -verify-diagnostics | FileCheck %s --check-prefix=FP

// G010: Intervening memory footprint near L1 size.
// Without footprint analysis: buffer fits in L1 (4cy load).
// With footprint analysis: intervening 32KB loop evicts the buffer, effective
// latency pushed to L2 (12cy). ALU=1 (cheap add), still RECOMPUTE either way,
// but the cost model reports different latencies.

module {
  func.func @footprint_evict(%x: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : i32

    // expected-remark @below {{cost-model: RECOMPUTE}}
    %buf = memref.alloc() : memref<1xi32>
    %other = memref.alloc() : memref<8192xi32>

    %val = arith.addi %x, %c1 : i32
    memref.store %val, %buf[%c0] : memref<1xi32>

    // Large intervening loop: touches 32KB (8192 x 4 bytes).
    affine.for %i = 0 to 8192 {
      memref.store %c1, %other[%i] : memref<8192xi32>
    }

    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %out = memref.load %buf[%c0] : memref<1xi32>
    memref.dealloc %other : memref<8192xi32>
    memref.dealloc %buf : memref<1xi32>
    return %out : i32
  }
}

// Without footprint: L1 hit, load replaced.
// NOFP-LABEL: func.func @footprint_evict
// NOFP-NOT:     memref.load %{{.*}} : memref<1xi32>
// NOFP:         return

// With footprint: eviction detected, still recompute (cheap ALU), load replaced.
// FP-LABEL: func.func @footprint_evict
// FP-NOT:     memref.load %{{.*}} : memref<1xi32>
// FP:         return
