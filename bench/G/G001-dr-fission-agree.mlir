// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-test-diagnostics})' -verify-diagnostics | FileCheck %s --check-prefix=DR
// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission{test-diagnostics})' | FileCheck %s --check-prefix=FISSION

// G001: Both DR and fission agree that recomputation is profitable.
// Cheap add chain (1 cycle), small buffer with 1 consumer.
// Data-recomputation says RECOMPUTE (cheap ALU, small buffer).
// Memory-fission has no sibling loops to fission, so it is a no-op.

module {
  func.func @agree(%x: memref<?xf64>, %n: index) -> f64 {
    %c0 = arith.constant 0 : index
    %cst0 = arith.constant 0.0 : f64
    %cst1 = arith.constant 1.0 : f64

    // expected-remark @below {{cost-model: RECOMPUTE}}
    %buf = memref.alloc() : memref<1xf64>

    %val = arith.addf %cst0, %cst1 : f64
    memref.store %val, %buf[%c0] : memref<1xf64>

    // expected-remark @below {{direct-forward: ACCEPT}}
    // expected-remark @below {{load: SINGLE}}
    %v = memref.load %buf[%c0] : memref<1xf64>

    memref.dealloc %buf : memref<1xf64>
    return %v : f64
  }
}

// Cost model says recompute, load is replaced.
// DR-LABEL: func.func @agree
// DR-NOT:     memref.load
// DR:         return

// No sibling loops to fission, pass is a no-op.
// FISSION-LABEL: func.func @agree
// FISSION:       memref.store
// FISSION:       memref.load
