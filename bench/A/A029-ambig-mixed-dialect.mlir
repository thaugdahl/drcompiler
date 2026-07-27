// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' --allow-unregistered-dialect -verify-diagnostics

// A029: store via polygeist.memref2pointer + llvm.store, load via memref.load.
// The pass must trace through the pointer conversion to classify correctly.

module {
  func.func private @write_via_ptr(%out: memref<i32>) {
    %ptr = "polygeist.memref2pointer"(%out) : (memref<i32>) -> !llvm.ptr
    %c99 = arith.constant 99 : i32
    llvm.store %c99, %ptr : i32, !llvm.ptr
    return
  }

  func.func @ambig_mixed_dialect() -> i32 {
    %alloc = memref.alloc() : memref<i32>
    %c42 = arith.constant 42 : i32
    memref.store %c42, %alloc[] : memref<i32>
    call @write_via_ptr(%alloc) : (memref<i32>) -> ()
    // expected-remark @below {{load: MULTI}}
    %v = memref.load %alloc[] : memref<i32>
    memref.dealloc %alloc : memref<i32>
    return %v : i32
  }
}
