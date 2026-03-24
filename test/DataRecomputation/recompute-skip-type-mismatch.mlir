// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute})' | FileCheck %s

// A SINGLE-provenance load whose type differs from the stored value must NOT be
// replaced.  This models byte-level reinterpretation patterns like endianness
// detection: store an i32, then load the first byte as i8 via the same alloca.

module {
  func.func @endian_check() -> i32 {
    %sz = llvm.mlir.constant(1 : i64) : i64
    %alloca = llvm.alloca %sz x i32 : (i64) -> !llvm.ptr
    %c1 = arith.constant 1 : i32
    llvm.store %c1, %alloca : i32, !llvm.ptr
    // Load the first byte — type differs from the stored i32.
    %byte = llvm.load %alloca : !llvm.ptr -> i8
    %ext = arith.extui %byte : i8 to i32
    return %ext : i32
  }
}

// The i8 load must survive — replacing it with the i32 stored value would make
// the arith.extui invalid (i32 → i32).
// CHECK-LABEL: func.func @endian_check
// CHECK:         llvm.load
// CHECK:         arith.extui
