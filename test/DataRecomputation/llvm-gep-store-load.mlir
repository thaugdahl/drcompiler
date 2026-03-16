// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// llvm.getelementptr + llvm.store/load: traces through GEP to alloca root.

module {
  func.func @test() {
    %c42 = arith.constant 42 : i32
    %c0 = arith.constant 0 : i32
    %sz = llvm.mlir.constant(128 : i64) : i64
    %alloca = llvm.alloca %sz x i32 : (i64) -> !llvm.ptr
    %gep = llvm.getelementptr %alloca[%c0] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    llvm.store %c42, %gep : i32, !llvm.ptr
    // expected-remark @below {{load: SINGLE}}
    %val = llvm.load %gep : !llvm.ptr -> i32
    return
  }
}
