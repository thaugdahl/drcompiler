// RUN: dr-opt --data-recomputation --dr-test-diagnostics %s 2>&1 | FileCheck %s

func.func @simple_store_load() {
  %alloc = memref.alloc() : memref<f32>
  %c0 = arith.constant 0.0 : f32
  memref.store %c0, %alloc[] : memref<f32>
  // expected-remark @below {{load: SINGLE}}
  %v = memref.load %alloc[] : memref<f32>
  memref.dealloc %alloc : memref<f32>
  return
}

func.func @branched_store(%cond: i1) {
  %alloc = memref.alloc() : memref<f32>
  %c1 = arith.constant 1.0 : f32
  %c2 = arith.constant 2.0 : f32
  scf.if %cond {
    memref.store %c1, %alloc[] : memref<f32>
  } else {
    memref.store %c2, %alloc[] : memref<f32>
  }
  // expected-remark @below {{load: MULTI}}
  %v = memref.load %alloc[] : memref<f32>
  memref.dealloc %alloc : memref<f32>
  return
}
