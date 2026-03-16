// RUN: dr-opt %s --allow-unregistered-dialect --raise-malloc-to-memref | FileCheck %s

// Simple malloc + direct store/load (index 0) rewritten to memref.

module {
  func.func private @malloc(i64) -> memref<?xi8>
  func.func private @free(!llvm.ptr)

  func.func @test() -> i32 {
    %c42 = arith.constant 42 : i32
    %c4 = arith.constant 4 : i64
    %raw = call @malloc(%c4) : (i64) -> memref<?xi8>
    %ptr = "polygeist.memref2pointer"(%raw) : (memref<?xi8>) -> !llvm.ptr
    llvm.store %c42, %ptr : i32, !llvm.ptr
    %val = llvm.load %ptr : !llvm.ptr -> i32
    call @free(%ptr) : (!llvm.ptr) -> ()
    return %val : i32
  }
}

// CHECK-LABEL: func.func @test
// CHECK:         %[[ALLOC:.*]] = memref.alloc() : memref<1xi32>
// CHECK:         memref.store %{{.*}}, %[[ALLOC]][%{{.*}}] : memref<1xi32>
// CHECK:         %[[VAL:.*]] = memref.load %[[ALLOC]][%{{.*}}] : memref<1xi32>
// CHECK:         memref.dealloc %[[ALLOC]]
// CHECK-NOT:     llvm.store
// CHECK-NOT:     llvm.load
// CHECK:         return %[[VAL]] : i32
