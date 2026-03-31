// RUN: dr-opt %s --allow-unregistered-dialect --raise-malloc-to-memref | FileCheck %s

// pointer2memref with memref.load/store uses (non-pointer uses).
// Phase 5 should lower these to llvm.getelementptr + llvm.load/store.

module {
  // --- Test 1: memref.load from pointer2memref ---
  func.func @load_from_ptr(%arg0: !llvm.ptr, %idx: index) -> i64 {
    %m = "polygeist.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xi64>
    %v = memref.load %m[%idx] : memref<?xi64>
    return %v : i64
  }

  // CHECK-LABEL: func.func @load_from_ptr
  // CHECK-SAME:    (%[[PTR:.*]]: !llvm.ptr, %[[IDX:.*]]: index)
  // CHECK:         %[[I64:.*]] = arith.index_cast %[[IDX]] : index to i64
  // CHECK:         %[[GEP:.*]] = llvm.getelementptr %[[PTR]][%[[I64]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  // CHECK:         %[[VAL:.*]] = llvm.load %[[GEP]] : !llvm.ptr -> i64
  // CHECK:         return %[[VAL]] : i64
  // CHECK-NOT:     polygeist.pointer2memref
  // CHECK-NOT:     memref.load

  // --- Test 2: memref.store to pointer2memref ---
  func.func @store_to_ptr(%arg0: !llvm.ptr, %val: i32, %idx: index) {
    %m = "polygeist.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xi32>
    memref.store %val, %m[%idx] : memref<?xi32>
    return
  }

  // CHECK-LABEL: func.func @store_to_ptr
  // CHECK-SAME:    (%[[PTR:.*]]: !llvm.ptr, %[[VAL:.*]]: i32, %[[IDX:.*]]: index)
  // CHECK:         %[[I64:.*]] = arith.index_cast %[[IDX]] : index to i64
  // CHECK:         %[[GEP:.*]] = llvm.getelementptr %[[PTR]][%[[I64]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
  // CHECK:         llvm.store %[[VAL]], %[[GEP]] : i32, !llvm.ptr
  // CHECK-NOT:     polygeist.pointer2memref
  // CHECK-NOT:     memref.store

  // --- Test 3: mixed uses (memref.load + memref2pointer, both lowered) ---
  func.func private @use_ptr(!llvm.ptr)

  func.func @mixed_uses(%arg0: !llvm.ptr, %idx: index) -> i64 {
    %m = "polygeist.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xi64>
    %v = memref.load %m[%idx] : memref<?xi64>
    %p = "polygeist.memref2pointer"(%m) : (memref<?xi64>) -> !llvm.ptr
    call @use_ptr(%p) : (!llvm.ptr) -> ()
    return %v : i64
  }

  // CHECK-LABEL: func.func @mixed_uses
  // CHECK-SAME:    (%[[PTR:.*]]: !llvm.ptr, %[[IDX:.*]]: index)
  // CHECK:         llvm.getelementptr
  // CHECK:         llvm.load
  // CHECK:         call @use_ptr
  // CHECK-NOT:     polygeist.pointer2memref
  // CHECK-NOT:     polygeist.memref2pointer

  // --- Test 4: scalar (no index) load from pointer2memref ---
  func.func @scalar_load(%arg0: !llvm.ptr) -> f64 {
    %m = "polygeist.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<f64>
    %v = memref.load %m[] : memref<f64>
    return %v : f64
  }

  // CHECK-LABEL: func.func @scalar_load
  // CHECK-SAME:    (%[[PTR:.*]]: !llvm.ptr)
  // CHECK:         %[[VAL:.*]] = llvm.load %[[PTR]] : !llvm.ptr -> f64
  // CHECK:         return %[[VAL]] : f64
  // CHECK-NOT:     polygeist.pointer2memref

  // --- Test 5: multiple loads and stores from same pointer2memref ---
  func.func @multi_access(%arg0: !llvm.ptr, %i: index, %j: index, %val: i32) -> i32 {
    %m = "polygeist.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xi32>
    memref.store %val, %m[%i] : memref<?xi32>
    %v = memref.load %m[%j] : memref<?xi32>
    return %v : i32
  }

  // CHECK-LABEL: func.func @multi_access
  // CHECK-SAME:    (%[[PTR:.*]]: !llvm.ptr, %[[I:.*]]: index, %[[J:.*]]: index, %[[VAL:.*]]: i32)
  // CHECK-DAG:     %[[I64I:.*]] = arith.index_cast %[[I]] : index to i64
  // CHECK-DAG:     %[[GEPI:.*]] = llvm.getelementptr %[[PTR]][%[[I64I]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
  // CHECK:         llvm.store %[[VAL]], %[[GEPI]] : i32, !llvm.ptr
  // CHECK:         %[[I64J:.*]] = arith.index_cast %[[J]] : index to i64
  // CHECK:         %[[GEPJ:.*]] = llvm.getelementptr %[[PTR]][%[[I64J]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
  // CHECK:         %[[LOADED:.*]] = llvm.load %[[GEPJ]] : !llvm.ptr -> i32
  // CHECK:         return %[[LOADED]] : i32
  // CHECK-NOT:     polygeist.pointer2memref
}
