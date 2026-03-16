// RUN: dr-opt %s --allow-unregistered-dialect --pass-pipeline='builtin.module(raise-malloc-to-memref,data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// End-to-end: raise-malloc produces memref ops, DataRecomputation analyzes them.

module {
  func.func private @malloc(i64) -> memref<?xi8>
  func.func private @free(!llvm.ptr)

  func.func @test() {
    %c42 = arith.constant 42 : i32
    %c4 = arith.constant 4 : i64
    %raw = call @malloc(%c4) : (i64) -> memref<?xi8>
    %ptr = "polygeist.memref2pointer"(%raw) : (memref<?xi8>) -> !llvm.ptr
    llvm.store %c42, %ptr : i32, !llvm.ptr
    // expected-remark @below {{load: SINGLE}}
    %val = llvm.load %ptr : !llvm.ptr -> i32
    call @free(%ptr) : (!llvm.ptr) -> ()
    return
  }
}
