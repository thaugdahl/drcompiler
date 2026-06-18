// RUN: dr-opt %s | dr-opt | FileCheck %s

// The `par` dialect parses and prints (M2 ops).

// CHECK-LABEL: func.func @rt
func.func @rt(%arg0: memref<128xf32>, %cst: f32) {
  // CHECK:      par.region {
  // CHECK-NEXT:   par.forall([0], [128], [1]) {
  // CHECK-NEXT:   ^bb0(%[[I:.*]]: index):
  // CHECK-NEXT:     memref.store %{{.*}}, %{{.*}}[%[[I]]] : memref<128xf32>
  // CHECK-NEXT:     par.yield
  // CHECK:        par.yield
  par.region {
    par.forall([0], [128], [1]) {
    ^bb0(%i: index):
      memref.store %cst, %arg0[%i] : memref<128xf32>
      par.yield
    }
    par.yield
  }
  return
}
