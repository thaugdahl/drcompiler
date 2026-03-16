// RUN: dr-opt %s --allow-unregistered-dialect --raise-malloc-to-memref | FileCheck %s

// Static-size malloc: 512 bytes = 128 x i32.

module {
  func.func private @malloc(i64) -> memref<?xi8>
  func.func private @free(!llvm.ptr)

  func.func @test() -> i32 {
    %c42 = arith.constant 42 : i32
    %c512 = arith.constant 512 : i64
    %raw = call @malloc(%c512) : (i64) -> memref<?xi8>
    %ptr = "polygeist.memref2pointer"(%raw) : (memref<?xi8>) -> !llvm.ptr
    %c5 = arith.constant 5 : i32
    %gep = llvm.getelementptr %ptr[%c5] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    llvm.store %c42, %gep : i32, !llvm.ptr
    %val = llvm.load %gep : !llvm.ptr -> i32
    call @free(%ptr) : (!llvm.ptr) -> ()
    return %val : i32
  }
}

// CHECK-LABEL: func.func @test
// CHECK:         %[[ALLOC:.*]] = memref.alloc() : memref<128xi32>
// CHECK:         memref.store %{{.*}}, %[[ALLOC]][%{{.*}}] : memref<128xi32>
// CHECK:         memref.load %[[ALLOC]][%{{.*}}] : memref<128xi32>
// CHECK:         memref.dealloc
