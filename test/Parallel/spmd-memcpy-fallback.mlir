// lower-krnl-global multi-D copy fallback: a krnl.memcpy whose length is NOT a
// compile-time constant (or whose offset does not delinearize against the
// memref strides) cannot become a multi-D affine copy, so the known-correct
// flat scf.for { memref.load; memref.store } over 1-D reinterpret_cast views is
// kept.  Such a band stays par.critical -- sound (the rewrite only ever changes
// the copy's loop form when it can prove the shard disjoint).  See §11.20.

// RUN: dr-opt %s -allow-unregistered-dialect --pass-pipeline='builtin.module(lower-krnl-global)' | FileCheck %s

// CHECK-LABEL: func.func @dyn_copy
// CHECK:         memref.reinterpret_cast
// CHECK:         scf.for
// CHECK:           memref.load
// CHECK:           memref.store
func.func @dyn_copy(%dst: memref<4x8xf32>, %src: memref<4x8xf32>, %n: i64) {
  "krnl.memcpy"(%dst, %src, %n) : (memref<4x8xf32>, memref<4x8xf32>, i64) -> ()
  return
}
