// 4 fp args, 16 cross-products, all live before stores.
func.func @p04(%A: memref<16xf32>, %a: f32, %b: f32, %c: f32, %d: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %c8 = arith.constant 8 : index
  %c9 = arith.constant 9 : index
  %c10 = arith.constant 10 : index
  %c11 = arith.constant 11 : index
  %c12 = arith.constant 12 : index
  %c13 = arith.constant 13 : index
  %c14 = arith.constant 14 : index
  %c15 = arith.constant 15 : index
  %v00 = arith.mulf %a, %a : f32
  %v01 = arith.mulf %a, %b : f32
  %v02 = arith.mulf %a, %c : f32
  %v03 = arith.mulf %a, %d : f32
  %v04 = arith.mulf %b, %a : f32
  %v05 = arith.mulf %b, %b : f32
  %v06 = arith.mulf %b, %c : f32
  %v07 = arith.mulf %b, %d : f32
  %v08 = arith.mulf %c, %a : f32
  %v09 = arith.mulf %c, %b : f32
  %v10 = arith.mulf %c, %c : f32
  %v11 = arith.mulf %c, %d : f32
  %v12 = arith.mulf %d, %a : f32
  %v13 = arith.mulf %d, %b : f32
  %v14 = arith.mulf %d, %c : f32
  %v15 = arith.mulf %d, %d : f32
  memref.store %v00, %A[%c0] : memref<16xf32>
  memref.store %v01, %A[%c1] : memref<16xf32>
  memref.store %v02, %A[%c2] : memref<16xf32>
  memref.store %v03, %A[%c3] : memref<16xf32>
  memref.store %v04, %A[%c4] : memref<16xf32>
  memref.store %v05, %A[%c5] : memref<16xf32>
  memref.store %v06, %A[%c6] : memref<16xf32>
  memref.store %v07, %A[%c7] : memref<16xf32>
  memref.store %v08, %A[%c8] : memref<16xf32>
  memref.store %v09, %A[%c9] : memref<16xf32>
  memref.store %v10, %A[%c10] : memref<16xf32>
  memref.store %v11, %A[%c11] : memref<16xf32>
  memref.store %v12, %A[%c12] : memref<16xf32>
  memref.store %v13, %A[%c13] : memref<16xf32>
  memref.store %v14, %A[%c14] : memref<16xf32>
  memref.store %v15, %A[%c15] : memref<16xf32>
  return
}
