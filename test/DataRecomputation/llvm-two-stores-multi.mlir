// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// Two llvm.stores to different GEP offsets of same alloca, load from one:
// MULTI because flat coverage (nullopt) can't distinguish offsets.

module {
  func.func @test() {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c0i = arith.constant 0 : i32
    %c1i = arith.constant 1 : i32
    %sz = llvm.mlir.constant(128 : i64) : i64
    %alloca = llvm.alloca %sz x i32 : (i64) -> !llvm.ptr
    %gep0 = llvm.getelementptr %alloca[%c0i] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    %gep1 = llvm.getelementptr %alloca[%c1i] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    llvm.store %c1, %gep0 : i32, !llvm.ptr
    llvm.store %c2, %gep1 : i32, !llvm.ptr
    // expected-remark @below {{load: MULTI}}
    %val = llvm.load %gep0 : !llvm.ptr -> i32
    return
  }
}
