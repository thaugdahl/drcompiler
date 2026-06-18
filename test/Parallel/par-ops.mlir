// RUN: dr-opt %s | dr-opt | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: dr-opt %s --pass-pipeline='builtin.module(func.func(convert-par-to-scf))' | FileCheck %s --check-prefix=LOWER

// par.barrier / par.redistribute / par.critical round-trip, and lower under the
// sequential sink (barrier + redistribute erased; critical inlined in order).

// ROUNDTRIP-LABEL: func.func @ops
// ROUNDTRIP:         par.region {
// ROUNDTRIP:           par.barrier
// ROUNDTRIP:           par.redistribute %{{.*}} : memref<128xf32> from "block:0" to "block:1"
// ROUNDTRIP:           par.critical {
// ROUNDTRIP:             memref.store
// ROUNDTRIP:             par.yield
// ROUNDTRIP:           }

// LOWER-LABEL: func.func @ops
// LOWER:         memref.store
// LOWER-NOT:     par.
func.func @ops(%X: memref<128xf32>, %z: f32, %c0: index) {
  par.region {
    par.barrier
    par.redistribute %X : memref<128xf32> from "block:0" to "block:1"
    par.critical {
      memref.store %z, %X[%c0] : memref<128xf32>
      par.yield
    }
    par.yield
  }
  return
}
