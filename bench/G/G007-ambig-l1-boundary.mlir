// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-l1-size=32768})' | FileCheck %s --check-prefix=DEFAULT
// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-l1-size=30000})' | FileCheck %s --check-prefix=SMALL

// G007: Buffer at ~31KB, ALU=1 (add), 1 consumer.
// With default L1=32KB: buffer fits L1 (4cy). keepCost=6, recomputeCost=1 → RECOMPUTE.
// With L1=30KB: buffer spills to L2 (12cy). keepCost=14, recomputeCost=1 → RECOMPUTE (even more).
// Both recompute, but the load latency estimate differs — sensitivity test.

module {
  func.func @l1_boundary(%x: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : i32

    %buf = memref.alloc() : memref<7936xi32>

    affine.for %i = 0 to 7936 {
      %val = arith.addi %x, %c1 : i32
      affine.store %val, %buf[%i] : memref<7936xi32>
    }

    %out = memref.load %buf[%c0] : memref<7936xi32>
    memref.dealloc %buf : memref<7936xi32>
    return %out : i32
  }
}

// Both configs recompute — load replaced.
// DEFAULT-LABEL: func.func @l1_boundary
// DEFAULT-NOT:     memref.load
// DEFAULT:         return

// SMALL-LABEL: func.func @l1_boundary
// SMALL-NOT:     memref.load
// SMALL:         return
