// RUN: dr-opt %s --allow-unregistered-dialect --raise-malloc-to-memref | FileCheck %s

// Test lowering of polygeist.subindex ops:
//   (a) memref2pointer uses → lowered via memref.view + Phase 4 pointer extraction
//   (b) Non-pointer uses (func.call, scf.yield) → memref.view lowering

module {
  func.func private @use_ptr(!llvm.ptr)
  func.func private @use_memref(memref<?xi8>)
  func.func private @use_memref_and_ptr(!llvm.ptr, memref<?xi8>)

  // --- Test: subindex where all uses are memref2pointer ---
  // Phase 4 lowers memref2pointer before Phase 6, so Phase 6 sees the
  // subindex feeding extract_aligned_pointer_as_index (from Phase 4).
  // Phase 6 replaces the subindex with memref.view.
  // CHECK-LABEL: func.func @subindex_all_m2p
  // CHECK-SAME:    (%[[ARG:.*]]: memref<?xi8>)
  // CHECK:         %[[VIEW:.*]] = memref.view %[[ARG]]
  // CHECK:         memref.extract_aligned_pointer_as_index %[[VIEW]]
  // CHECK:         call @use_ptr(
  // CHECK-NOT:     polygeist.subindex
  func.func @subindex_all_m2p(%arg0: memref<?xi8>) {
    %c6 = arith.constant 6 : index
    %si = "polygeist.subindex"(%arg0, %c6) : (memref<?xi8>, index) -> memref<?xi8>
    %ptr = "polygeist.memref2pointer"(%si) : (memref<?xi8>) -> !llvm.ptr
    call @use_ptr(%ptr) : (!llvm.ptr) -> ()
    return
  }

  // --- Test: subindex passed to a function call (non-pointer use) ---
  // CHECK-LABEL: func.func @subindex_func_call
  // CHECK-SAME:    (%[[ARG:.*]]: memref<?xi8>)
  // CHECK:         %[[C6:.*]] = arith.constant 6 : index
  // CHECK:         %[[C0:.*]] = arith.constant 0 : index
  // CHECK:         %[[DIM:.*]] = memref.dim %[[ARG]], %[[C0]]
  // CHECK:         %[[REM:.*]] = arith.subi %[[DIM]], %[[C6]]
  // CHECK:         %[[VIEW:.*]] = memref.view %[[ARG]][%[[C6]]][%[[REM]]]
  // CHECK-SAME:      : memref<?xi8> to memref<?xi8>
  // CHECK:         call @use_memref(%[[VIEW]])
  // CHECK-NOT:     polygeist.subindex
  func.func @subindex_func_call(%arg0: memref<?xi8>) {
    %c6 = arith.constant 6 : index
    %si = "polygeist.subindex"(%arg0, %c6) : (memref<?xi8>, index) -> memref<?xi8>
    call @use_memref(%si) : (memref<?xi8>) -> ()
    return
  }

  // --- Test: subindex with mixed uses (memref2pointer + func.call) ---
  // CHECK-LABEL: func.func @subindex_mixed_uses
  // CHECK-SAME:    (%[[ARG:.*]]: memref<?xi8>)
  // CHECK:         %[[VIEW:.*]] = memref.view %[[ARG]]
  // CHECK:         memref.extract_aligned_pointer_as_index %[[VIEW]]
  // CHECK:         call @use_ptr(
  // CHECK:         call @use_memref(%[[VIEW]])
  // CHECK-NOT:     polygeist.subindex
  func.func @subindex_mixed_uses(%arg0: memref<?xi8>) {
    %c6 = arith.constant 6 : index
    %si = "polygeist.subindex"(%arg0, %c6) : (memref<?xi8>, index) -> memref<?xi8>
    %ptr = "polygeist.memref2pointer"(%si) : (memref<?xi8>) -> !llvm.ptr
    call @use_ptr(%ptr) : (!llvm.ptr) -> ()
    call @use_memref(%si) : (memref<?xi8>) -> ()
    return
  }

  // --- Test: subindex used in scf.yield (loop-carried) ---
  // CHECK-LABEL: func.func @subindex_scf_yield
  // CHECK-NOT:     polygeist.subindex
  // CHECK:         memref.view
  // CHECK:         scf.yield
  func.func @subindex_scf_yield(%arg0: memref<?xi8>, %n: index) -> memref<?xi8> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %result = scf.for %iv = %c0 to %n step %c1 iter_args(%buf = %arg0) -> memref<?xi8> {
      %si = "polygeist.subindex"(%buf, %c1) : (memref<?xi8>, index) -> memref<?xi8>
      scf.yield %si : memref<?xi8>
    }
    return %result : memref<?xi8>
  }
}
