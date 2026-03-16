// RUN: dr-opt %s --allow-unregistered-dialect --raise-malloc-to-memref | FileCheck %s

// Pointer escapes to a non-free call: pass should NOT rewrite.

module {
  func.func private @malloc(i64) -> memref<?xi8>
  func.func private @free(!llvm.ptr)
  llvm.func @sink(!llvm.ptr)

  func.func @test() {
    %c4 = arith.constant 4 : i64
    %raw = call @malloc(%c4) : (i64) -> memref<?xi8>
    %ptr = "polygeist.memref2pointer"(%raw) : (memref<?xi8>) -> !llvm.ptr
    llvm.call @sink(%ptr) : (!llvm.ptr) -> ()
    call @free(%ptr) : (!llvm.ptr) -> ()
    return
  }
}

// CHECK-LABEL: func.func @test
// CHECK:         call @malloc
// CHECK:         "polygeist.memref2pointer"
// CHECK:         llvm.call @sink
// CHECK:         call @free
