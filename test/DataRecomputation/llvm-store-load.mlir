// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// llvm.store followed by llvm.load on an alloca: SINGLE provenance.

module {
  func.func @test() {
    %c42 = arith.constant 42 : i32
    %one = llvm.mlir.constant(1 : i64) : i64
    %alloca = llvm.alloca %one x i32 : (i64) -> !llvm.ptr
    llvm.store %c42, %alloca : i32, !llvm.ptr
    // expected-remark @below {{load: SINGLE}}
    %val = llvm.load %alloca : !llvm.ptr -> i32
    return
  }
}
