// 6 fp args; build 21 sums (a triangular table) all live before stores.
func.func @p05(%A: memref<21xf32>, %a: f32, %b: f32, %c: f32, %d: f32,
                %e: f32, %f: f32) {
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
  %v00 = arith.addf %a, %a : f32
  %v01 = arith.addf %a, %b : f32
  %v02 = arith.addf %a, %c : f32
  %v03 = arith.addf %a, %d : f32
  %v04 = arith.addf %a, %e : f32
  %v05 = arith.addf %a, %f : f32
  %v06 = arith.addf %b, %b : f32
  %v07 = arith.addf %b, %c : f32
  %v08 = arith.addf %b, %d : f32
  %v09 = arith.addf %b, %e : f32
  %v10 = arith.addf %b, %f : f32
  %v11 = arith.addf %c, %c : f32
  %v12 = arith.addf %c, %d : f32
  %v13 = arith.addf %c, %e : f32
  %v14 = arith.addf %c, %f : f32
  %v15 = arith.addf %d, %d : f32
  %v16 = arith.addf %d, %e : f32
  %v17 = arith.addf %d, %f : f32
  %v18 = arith.addf %e, %e : f32
  %v19 = arith.addf %e, %f : f32
  %v20 = arith.addf %f, %f : f32
  memref.store %v00, %A[%c0] : memref<21xf32>
  memref.store %v01, %A[%c1] : memref<21xf32>
  memref.store %v02, %A[%c2] : memref<21xf32>
  memref.store %v03, %A[%c3] : memref<21xf32>
  memref.store %v04, %A[%c4] : memref<21xf32>
  memref.store %v05, %A[%c5] : memref<21xf32>
  memref.store %v06, %A[%c6] : memref<21xf32>
  memref.store %v07, %A[%c7] : memref<21xf32>
  memref.store %v08, %A[%c8] : memref<21xf32>
  memref.store %v09, %A[%c9] : memref<21xf32>
  memref.store %v10, %A[%c10] : memref<21xf32>
  memref.store %v11, %A[%c11] : memref<21xf32>
  memref.store %v12, %A[%c12] : memref<21xf32>
  memref.store %v13, %A[%c13] : memref<21xf32>
  memref.store %v14, %A[%c14] : memref<21xf32>
  memref.store %v15, %A[%c15] : memref<21xf32>
  memref.store %v16, %A[%c16] : memref<21xf32>
  memref.store %v17, %A[%c17] : memref<21xf32>
  memref.store %v18, %A[%c18] : memref<21xf32>
  memref.store %v19, %A[%c19] : memref<21xf32>
  memref.store %v20, %A[%c20] : memref<21xf32>
  return
}
