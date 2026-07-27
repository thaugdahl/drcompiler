// RUN: dr-opt %s --allow-unregistered-dialect --pass-pipeline='builtin.module(data-recomputation{dr-recompute=true dr-buffer-elim=true dr-summary=true})' 2>&1 | FileCheck %s --check-prefix=SUM

// E005: Access through a memref.subview chain. The escape analysis must
// chase view-like ops and still recognize the buffer as non-escaping.

// SUM: buffer-elim {{.*}}: FEASIBLE (escape=no-escape

func.func @view_chain() {
  %a = memref.alloca() : memref<4xi32>
  %sv = memref.subview %a[1][2][1] : memref<4xi32> to memref<2xi32, strided<[1], offset: 1>>
  %c = arith.constant 55 : i32
  %idx0 = arith.constant 0 : index
  affine.store %c, %a[0] : memref<4xi32>
  affine.store %c, %a[1] : memref<4xi32>
  affine.store %c, %a[2] : memref<4xi32>
  affine.store %c, %a[3] : memref<4xi32>
  %v0 = affine.load %a[0] : memref<4xi32>
  %v1 = memref.load %sv[%idx0] : memref<2xi32, strided<[1], offset: 1>>
  "use.consume"(%v0, %v1) : (i32, i32) -> ()
  return
}
