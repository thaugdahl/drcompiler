// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute})' | FileCheck %s

// Interprocedural rematerialization through affine ops.
//
// produce() stores x*7+13 into buf[0] via affine.store.
// main() calls produce(), then reads buf[0] via affine.load.
// The pass should trace the SINGLE provenance interprocedurally,
// clone the pure computation (x*7+13) into main(), and eliminate
// the affine.load.
//
// Derived from: cgeist -O0 --raise-scf-to-affine on:
//   void produce(int *buf, int x) { buf[0] = x * 7 + 13; }
//   int main(int argc, char **argv) {
//       int buf[1]; produce(buf, argc); return buf[0];
//   }

module {
  func.func @produce(%buf: memref<?xi32>, %x: i32) {
    %c7 = arith.constant 7 : i32
    %c13 = arith.constant 13 : i32
    %0 = arith.muli %x, %c7 : i32
    %1 = arith.addi %0, %c13 : i32
    affine.store %1, %buf[0] : memref<?xi32>
    return
  }

  func.func @main(%argc: i32) -> i32 {
    %alloca = memref.alloca() : memref<1xi32>
    %cast = memref.cast %alloca : memref<1xi32> to memref<?xi32>
    call @produce(%cast, %argc) : (memref<?xi32>, i32) -> ()
    %0 = affine.load %alloca[0] : memref<1xi32>
    return %0 : i32
  }
}

// CHECK-LABEL: func.func @main
// CHECK:         call @produce
// CHECK-NOT:     affine.load
// CHECK:         %[[C7:.*]] = arith.constant 7 : i32
// CHECK:         %[[MUL:.*]] = arith.muli %{{.*}}, %[[C7]] : i32
// CHECK:         %[[C13:.*]] = arith.constant 13 : i32
// CHECK:         %[[ADD:.*]] = arith.addi %[[MUL]], %[[C13]] : i32
// CHECK:         return %[[ADD]] : i32
