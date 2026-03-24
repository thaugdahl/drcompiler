// RUN: dr-opt %s --allow-unregistered-dialect --raise-malloc-to-memref | FileCheck %s

// Residual polygeist.memref2pointer ops (not part of a malloc group) should be
// lowered to memref.extract_aligned_pointer_as_index + arith.index_cast + llvm.inttoptr.

module {
  func.func private @use_ptr(!llvm.ptr)

  func.func @non_malloc_memref2ptr(%arg0: memref<?xi8>) {
    %ptr = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
    call @use_ptr(%ptr) : (!llvm.ptr) -> ()
    return
  }
}

// CHECK-LABEL: func.func @non_malloc_memref2ptr
// CHECK-SAME:    (%[[ARG:.*]]: memref<?xi8>)
// CHECK:         %[[IDX:.*]] = memref.extract_aligned_pointer_as_index %[[ARG]]
// CHECK:         %[[I64:.*]] = arith.index_cast %[[IDX]] : index to i64
// CHECK:         %[[PTR:.*]] = llvm.inttoptr %[[I64]] : i64 to !llvm.ptr
// CHECK:         call @use_ptr(%[[PTR]])
// CHECK-NOT:     polygeist.memref2pointer
