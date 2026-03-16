// RUN: dr-opt %s --allow-unregistered-dialect --raise-malloc-to-memref | FileCheck %s

// malloc with GEP-indexed store/load rewritten to memref ops.

module {
  func.func private @malloc(i64) -> memref<?xi8>
  func.func private @free(!llvm.ptr)

  func.func @test(%n: i32) {
    %c4 = arith.constant 4 : i64
    %n_ext = arith.extsi %n : i32 to i64
    %bytes = arith.muli %n_ext, %c4 : i64
    %raw = call @malloc(%bytes) : (i64) -> memref<?xi8>
    %ptr = "polygeist.memref2pointer"(%raw) : (memref<?xi8>) -> !llvm.ptr
    %c0 = arith.constant 0 : i32
    %gep = llvm.getelementptr %ptr[%c0] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    %c42 = arith.constant 42 : i32
    llvm.store %c42, %gep : i32, !llvm.ptr
    %val = llvm.load %gep : !llvm.ptr -> i32
    call @free(%ptr) : (!llvm.ptr) -> ()
    return
  }
}

// CHECK-LABEL: func.func @test
// CHECK:         %[[ALLOC:.*]] = memref.alloc(%{{.*}}) : memref<?xi32>
// CHECK:         memref.store
// CHECK:         memref.load
// CHECK:         memref.dealloc
// CHECK-NOT:     llvm.getelementptr
// CHECK-NOT:     llvm.store
// CHECK-NOT:     llvm.load
