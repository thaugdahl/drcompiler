// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' --allow-unregistered-dialect -verify-diagnostics

// S1-C: defined callee converts its memref arg to !llvm.ptr via
// polygeist.memref2pointer and writes through the pointer.  The pass must
// trace through that conversion in analyzeCalleeArg and classify the
// callee as a writer; otherwise the caller's pre-call store reaches the
// post-call load as SINGLE provenance and gets eliminated -- producing a
// stale value.

module {
  func.func private @write_via_ptr(%out: memref<i32>) {
    %ptr = "polygeist.memref2pointer"(%out) : (memref<i32>) -> !llvm.ptr
    %c99 = arith.constant 99 : i32
    llvm.store %c99, %ptr : i32, !llvm.ptr
    return
  }

  func.func @test() -> i32 {
    %alloc = memref.alloc() : memref<i32>
    %c42 = arith.constant 42 : i32
    memref.store %c42, %alloc[] : memref<i32>
    func.call @write_via_ptr(%alloc) : (memref<i32>) -> ()
    // expected-remark @below {{load: MULTI}}
    %v = memref.load %alloc[] : memref<i32>
    memref.dealloc %alloc : memref<i32>
    return %v : i32
  }
}
