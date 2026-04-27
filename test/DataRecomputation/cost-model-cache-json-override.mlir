// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics cpu-cost-model-file=%S/Inputs/cache-override.json})' -verify-diagnostics | FileCheck %s

// S3-A: the cost-model JSON file supplies a `cache` object that overrides
// the CLI / built-in cache parameters.  The JSON sets l1_size=1024,
// l2_size=4096, l3_size=0, l1_latency=7, l2_latency=13, mem_latency=251.
// A 32 KB buffer is bigger than l2_size and l3 is unmodeled, so the
// reported load latency must be 251 (mem_latency from the file), not the
// CLI default of 200.

module {
  func.func @run(%x: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1 : i32

    // 8192 elements × 4 bytes = 32768 bytes (> l2_size=4096, l3_size=0).
    // expected-remark @+1 {{cost-model: RECOMPUTE (compute=1, load=251}}
    %buf = memref.alloc() : memref<8192xi32>

    affine.for %i = 0 to 8192 {
      %val = arith.addi %x, %one : i32
      affine.store %val, %buf[%i] : memref<8192xi32>
    }

    // expected-remark @+1 {{load: SINGLE}}
    %v = memref.load %buf[%c0] : memref<8192xi32>

    memref.dealloc %buf : memref<8192xi32>
    return %v : i32
  }
}

// CHECK-LABEL: func.func @run
