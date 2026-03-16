// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// llvm.store in both branches of scf.if, then llvm.load: MULTI.

module {
  func.func @test(%cond: i1) {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %one = llvm.mlir.constant(1 : i64) : i64
    %alloca = llvm.alloca %one x i32 : (i64) -> !llvm.ptr
    scf.if %cond {
      llvm.store %c1, %alloca : i32, !llvm.ptr
    } else {
      llvm.store %c2, %alloca : i32, !llvm.ptr
    }
    // expected-remark @below {{load: MULTI}}
    %val = llvm.load %alloca : !llvm.ptr -> i32
    return
  }
}
