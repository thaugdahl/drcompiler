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

// Malloc group is not rewritten (pointer escapes), but the residual
// memref2pointer is lowered to extract_aligned_pointer_as_index.
// CHECK-LABEL: func.func @test
// CHECK:         call @malloc
// CHECK:         memref.extract_aligned_pointer_as_index
// CHECK:         arith.index_cast
// CHECK:         llvm.inttoptr
// CHECK:         llvm.call @sink
// CHECK:         call @free
