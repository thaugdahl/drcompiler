// RUN: dr-opt %s --allow-unregistered-dialect --raise-malloc-to-memref | FileCheck %s

// After struct-memref preprocessing, memref2pointer may have !llvm.ptr input
// (identity cast).  The pass should eliminate it.

module {
  llvm.func @sink(!llvm.ptr)

  func.func @identity_cast(%arg0: !llvm.ptr) {
    %ptr = "polygeist.memref2pointer"(%arg0) : (!llvm.ptr) -> !llvm.ptr
    llvm.call @sink(%ptr) : (!llvm.ptr) -> ()
    return
  }
}

// CHECK-LABEL: func.func @identity_cast
// CHECK-SAME:    (%[[ARG:.*]]: !llvm.ptr)
// CHECK-NEXT:    llvm.call @sink(%[[ARG]])
// CHECK-NOT:     polygeist.memref2pointer
