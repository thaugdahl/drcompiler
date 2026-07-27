// 5 args x 5 muls = 25 distinct products, then stored.  Forces all 25 live
// briefly before stores commit them.  Should spill on 16 GP archs.
func.func @p03(%A: memref<25xi32>, %a: i32, %b: i32, %c: i32, %d: i32, %e: i32) {
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
  %c16 = arith.constant 16 : index
  %c17 = arith.constant 17 : index
  %c18 = arith.constant 18 : index
  %c19 = arith.constant 19 : index
  %c20 = arith.constant 20 : index
  %c21 = arith.constant 21 : index
  %c22 = arith.constant 22 : index
  %c23 = arith.constant 23 : index
  %c24 = arith.constant 24 : index
  %v00 = arith.muli %a, %a : i32
  %v01 = arith.muli %a, %b : i32
  %v02 = arith.muli %a, %c : i32
  %v03 = arith.muli %a, %d : i32
  %v04 = arith.muli %a, %e : i32
  %v05 = arith.muli %b, %a : i32
  %v06 = arith.muli %b, %b : i32
  %v07 = arith.muli %b, %c : i32
  %v08 = arith.muli %b, %d : i32
  %v09 = arith.muli %b, %e : i32
  %v10 = arith.muli %c, %a : i32
  %v11 = arith.muli %c, %b : i32
  %v12 = arith.muli %c, %c : i32
  %v13 = arith.muli %c, %d : i32
  %v14 = arith.muli %c, %e : i32
  %v15 = arith.muli %d, %a : i32
  %v16 = arith.muli %d, %b : i32
  %v17 = arith.muli %d, %c : i32
  %v18 = arith.muli %d, %d : i32
  %v19 = arith.muli %d, %e : i32
  %v20 = arith.muli %e, %a : i32
  %v21 = arith.muli %e, %b : i32
  %v22 = arith.muli %e, %c : i32
  %v23 = arith.muli %e, %d : i32
  %v24 = arith.muli %e, %e : i32
  memref.store %v00, %A[%c0] : memref<25xi32>
  memref.store %v01, %A[%c1] : memref<25xi32>
  memref.store %v02, %A[%c2] : memref<25xi32>
  memref.store %v03, %A[%c3] : memref<25xi32>
  memref.store %v04, %A[%c4] : memref<25xi32>
  memref.store %v05, %A[%c5] : memref<25xi32>
  memref.store %v06, %A[%c6] : memref<25xi32>
  memref.store %v07, %A[%c7] : memref<25xi32>
  memref.store %v08, %A[%c8] : memref<25xi32>
  memref.store %v09, %A[%c9] : memref<25xi32>
  memref.store %v10, %A[%c10] : memref<25xi32>
  memref.store %v11, %A[%c11] : memref<25xi32>
  memref.store %v12, %A[%c12] : memref<25xi32>
  memref.store %v13, %A[%c13] : memref<25xi32>
  memref.store %v14, %A[%c14] : memref<25xi32>
  memref.store %v15, %A[%c15] : memref<25xi32>
  memref.store %v16, %A[%c16] : memref<25xi32>
  memref.store %v17, %A[%c17] : memref<25xi32>
  memref.store %v18, %A[%c18] : memref<25xi32>
  memref.store %v19, %A[%c19] : memref<25xi32>
  memref.store %v20, %A[%c20] : memref<25xi32>
  memref.store %v21, %A[%c21] : memref<25xi32>
  memref.store %v22, %A[%c22] : memref<25xi32>
  memref.store %v23, %A[%c23] : memref<25xi32>
  memref.store %v24, %A[%c24] : memref<25xi32>
  return
}
