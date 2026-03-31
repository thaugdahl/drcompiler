// RUN: dr-opt %s --allow-unregistered-dialect --raise-malloc-to-memref | FileCheck %s

// After the struct-memref text preprocessor rewrites
//   memref.alloca() : memref<1x!llvm.struct<...>>
// to
//   llvm.alloca %size x !llvm.struct<...> : (i64) -> !llvm.ptr
// the polygeist.memref2pointer becomes an identity cast (!llvm.ptr -> !llvm.ptr).
// The raise pass should eliminate the identity cast cleanly.

module {
  llvm.func @sink(!llvm.ptr)

  func.func @struct_alloca_use() {
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %alloca = llvm.alloca %c1 x !llvm.struct<(i32, i64)> : (i64) -> !llvm.ptr
    %ptr = "polygeist.memref2pointer"(%alloca) : (!llvm.ptr) -> !llvm.ptr
    llvm.call @sink(%ptr) : (!llvm.ptr) -> ()
    return
  }
}

// CHECK-LABEL: func.func @struct_alloca_use
// CHECK:         llvm.alloca
// CHECK-SAME:    !llvm.struct<(i32, i64)>
// CHECK:         llvm.call @sink
// CHECK-NOT:     polygeist.memref2pointer
