// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics dr-mem-latency=177 cpu-cost-model-file=%S/Inputs/cache-override.json})' -verify-diagnostics | FileCheck %s

// WP1 (COSTMODEL_V4_SPEC §2) precedence: an EXPLICIT CLI cache option wins over
// the cost-model JSON, which in turn wins over the built-in default.  The JSON
// fixture (cache-override.json) sets mem_latency=251; here the CLI also sets
// dr-mem-latency=177.  The resolved latency for a buffer that overflows L2
// (l3 unmodeled) must be 177 (CLI), NOT 251 (file) and NOT 200 (default).
//
// This is the regression guard for the old per-field jsonCache merge, which
// applied the file over the CLI value and would have reported load=251.

module {
  func.func @run(%x: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1 : i32

    // 8192 elements × 4 bytes = 32768 bytes (> l2_size=4096, l3_size=0).
    // expected-remark @below {{cost-model: RECOMPUTE (compute=1, load=177}}
    %buf = memref.alloc() : memref<8192xi32>

    affine.for %i = 0 to 8192 {
      %val = arith.addi %x, %one : i32
      affine.store %val, %buf[%i] : memref<8192xi32>
    }

    // expected-remark @below {{full-remat: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %buf[%c0] : memref<8192xi32>

    memref.dealloc %buf : memref<8192xi32>
    return %v : i32
  }
}

// CHECK-LABEL: func.func @run
