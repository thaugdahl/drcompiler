// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model})' | FileCheck %s --check-prefix=DEFAULT
// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model cpu-cost-model-file=%S/Inputs/keep-override.json})' | FileCheck %s --check-prefix=OVERRIDE

// G012: Default parameters say RECOMPUTE (cheap add, L1 buffer, 2 consumers).
// JSON override inflates add cost to 50 cycles and shrinks load latency to 2,
// flipping the decision to KEEP with 2 consumers.
// Default: keepCost=1+1+2*4=10, recompute=2*1=2 => RECOMPUTE.
// Override: keepCost=50+1+2*2=55, recompute=2*50=100 => KEEP.

module {
  func.func @json_override(%x: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : i32

    %buf = memref.alloc() : memref<1xi32>

    %val = arith.addi %x, %c1 : i32
    memref.store %val, %buf[%c0] : memref<1xi32>

    %a = memref.load %buf[%c0] : memref<1xi32>
    %b = memref.load %buf[%c0] : memref<1xi32>

    %r = arith.addi %a, %b : i32
    memref.dealloc %buf : memref<1xi32>
    return %r : i32
  }
}

// Default: RECOMPUTE, loads eliminated.
// DEFAULT-LABEL: func.func @json_override
// DEFAULT-NOT:     memref.load
// DEFAULT:         return

// Override: KEEP, loads survive.
// OVERRIDE-LABEL: func.func @json_override
// OVERRIDE:       memref.load
// OVERRIDE:       memref.load
// OVERRIDE:       return
