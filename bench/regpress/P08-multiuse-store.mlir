// Each value is used 2-3 times across non-adjacent operations, then stored.
// Forces LLVM to keep the values in registers across multiple uses.
func.func @p08(%A: memref<20xi32>, %a: i32, %b: i32, %c: i32, %d: i32, %e: i32) {
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
  // Phase A: 10 distinct products.
  %p0 = arith.muli %a, %b : i32
  %p1 = arith.muli %a, %c : i32
  %p2 = arith.muli %a, %d : i32
  %p3 = arith.muli %a, %e : i32
  %p4 = arith.muli %b, %c : i32
  %p5 = arith.muli %b, %d : i32
  %p6 = arith.muli %b, %e : i32
  %p7 = arith.muli %c, %d : i32
  %p8 = arith.muli %c, %e : i32
  %p9 = arith.muli %d, %e : i32
  // Phase B: cross sums using each p in multiple places.
  %s0 = arith.addi %p0, %p9 : i32
  %s1 = arith.addi %p1, %p8 : i32
  %s2 = arith.addi %p2, %p7 : i32
  %s3 = arith.addi %p3, %p6 : i32
  %s4 = arith.addi %p4, %p5 : i32
  %s5 = arith.addi %p0, %p5 : i32  // p0 reused
  %s6 = arith.addi %p1, %p6 : i32  // p1 reused
  %s7 = arith.addi %p2, %p7 : i32  // p2 reused
  %s8 = arith.addi %p3, %p8 : i32  // p3 reused
  %s9 = arith.addi %p4, %p9 : i32  // p4 reused
  // Stores at the end so all 20 values live across phase B.
  memref.store %p0, %A[%c0] : memref<20xi32>
  memref.store %p1, %A[%c1] : memref<20xi32>
  memref.store %p2, %A[%c2] : memref<20xi32>
  memref.store %p3, %A[%c3] : memref<20xi32>
  memref.store %p4, %A[%c4] : memref<20xi32>
  memref.store %p5, %A[%c5] : memref<20xi32>
  memref.store %p6, %A[%c6] : memref<20xi32>
  memref.store %p7, %A[%c7] : memref<20xi32>
  memref.store %p8, %A[%c8] : memref<20xi32>
  memref.store %p9, %A[%c9] : memref<20xi32>
  memref.store %s0, %A[%c10] : memref<20xi32>
  memref.store %s1, %A[%c11] : memref<20xi32>
  memref.store %s2, %A[%c12] : memref<20xi32>
  memref.store %s3, %A[%c13] : memref<20xi32>
  memref.store %s4, %A[%c14] : memref<20xi32>
  memref.store %s5, %A[%c15] : memref<20xi32>
  memref.store %s6, %A[%c16] : memref<20xi32>
  memref.store %s7, %A[%c17] : memref<20xi32>
  memref.store %s8, %A[%c18] : memref<20xi32>
  memref.store %s9, %A[%c19] : memref<20xi32>
  return
}
