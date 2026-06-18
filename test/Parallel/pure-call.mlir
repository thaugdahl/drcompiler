// RUN: dr-opt %s --pass-pipeline='builtin.module(dr-par-bubbles{par-test-diagnostics})' -verify-diagnostics
// RUN: dr-opt %s --pass-pipeline='builtin.module(dr-par-bubbles{par-materialize})' | FileCheck %s

// M4 (sound subset): a call to a provably-PURE function (touches no memory) is
// benign under parallel execution, so a loop whose only opaque op is such a
// call stays parallel and materializes.  Impure / external callees stay
// conservative.

func.func private @sq(%x: f32) -> f32 {
  %m = arith.mulf %x, %x : f32
  return %m : f32
}
memref.global "private" @g : memref<f32> = uninitialized
func.func private @sink(%x: f32) {
  %gg = memref.get_global @g : memref<f32>
  memref.store %x, %gg[] : memref<f32>
  return
}
func.func private @ext(%x: f32) -> f32

// CHECK-LABEL: func.func @pure_call
// CHECK:         par.forall([0], [128], [1]) {
// CHECK:           memref.load
// CHECK:           func.call @sq
// CHECK:           memref.store
// CHECK:           par.yield
func.func @pure_call() {
  %A = memref.alloc() : memref<128xf32>
  %B = memref.alloc() : memref<128xf32>
  // expected-remark @below {{par-bubble axis: PARALLEL}}
  affine.for %i = 0 to 128 {
    %a = affine.load %A[%i] : memref<128xf32>
    %s = func.call @sq(%a) : (f32) -> f32
    affine.store %s, %B[%i] : memref<128xf32>
  }
  return
}

// CHECK-LABEL: func.func @impure_call
// CHECK:         affine.for
// CHECK-NOT:     par.forall
func.func @impure_call() {
  %A = memref.alloc() : memref<128xf32>
  // expected-remark @below {{par-bubble axis: SEQUENTIAL (conservative)}}
  affine.for %i = 0 to 128 {
    %a = affine.load %A[%i] : memref<128xf32>
    func.call @sink(%a) : (f32) -> ()
  }
  return
}

// CHECK-LABEL: func.func @external_call
// CHECK:         affine.for
// CHECK-NOT:     par.forall
func.func @external_call() {
  %A = memref.alloc() : memref<128xf32>
  %B = memref.alloc() : memref<128xf32>
  // expected-remark @below {{par-bubble axis: SEQUENTIAL (conservative)}}
  affine.for %i = 0 to 128 {
    %a = affine.load %A[%i] : memref<128xf32>
    %s = func.call @ext(%a) : (f32) -> f32
    affine.store %s, %B[%i] : memref<128xf32>
  }
  return
}
