// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

func.func @single_scalar() {
  %m = memref.alloc() : memref<f32>
  %c = arith.constant 42.0 : f32
  memref.store %c, %m[] : memref<f32>
  // expected-remark @below {{load: SINGLE}}
  %v = memref.load %m[] : memref<f32>
  memref.dealloc %m : memref<f32>
  return
}
