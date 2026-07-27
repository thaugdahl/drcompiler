// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// F018: 8 chained memref.reinterpret_cast ops on the same alloc.
// Tests view-chain tracking depth.

module {
  func.func @chained_views() -> i32 {
    %alloc = memref.alloc() : memref<32xi32>
    %c42 = arith.constant 42 : i32
    %idx = arith.constant 0 : index

    // 8 chained reinterpret_cast ops, each from the previous.
    %v1 = memref.reinterpret_cast %alloc to offset: [0], sizes: [32], strides: [1]
        : memref<32xi32> to memref<32xi32, strided<[1], offset: 0>>
    %v2 = memref.reinterpret_cast %v1 to offset: [0], sizes: [32], strides: [1]
        : memref<32xi32, strided<[1], offset: 0>> to memref<32xi32, strided<[1], offset: 0>>
    %v3 = memref.reinterpret_cast %v2 to offset: [0], sizes: [32], strides: [1]
        : memref<32xi32, strided<[1], offset: 0>> to memref<32xi32, strided<[1], offset: 0>>
    %v4 = memref.reinterpret_cast %v3 to offset: [0], sizes: [32], strides: [1]
        : memref<32xi32, strided<[1], offset: 0>> to memref<32xi32, strided<[1], offset: 0>>
    %v5 = memref.reinterpret_cast %v4 to offset: [0], sizes: [32], strides: [1]
        : memref<32xi32, strided<[1], offset: 0>> to memref<32xi32, strided<[1], offset: 0>>
    %v6 = memref.reinterpret_cast %v5 to offset: [0], sizes: [32], strides: [1]
        : memref<32xi32, strided<[1], offset: 0>> to memref<32xi32, strided<[1], offset: 0>>
    %v7 = memref.reinterpret_cast %v6 to offset: [0], sizes: [32], strides: [1]
        : memref<32xi32, strided<[1], offset: 0>> to memref<32xi32, strided<[1], offset: 0>>
    %v8 = memref.reinterpret_cast %v7 to offset: [0], sizes: [32], strides: [1]
        : memref<32xi32, strided<[1], offset: 0>> to memref<32xi32, strided<[1], offset: 0>>

    // Store through the 8th view
    memref.store %c42, %v8[%idx] : memref<32xi32, strided<[1], offset: 0>>

    // Load from the original alloc — should see through 8 views
    // expected-remark @below {{load: SINGLE}}
    %out = memref.load %alloc[%idx] : memref<32xi32>

    memref.dealloc %alloc : memref<32xi32>
    return %out : i32
  }
}
