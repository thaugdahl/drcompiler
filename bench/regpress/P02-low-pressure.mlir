// 4 args, all computed, stored to memref to prevent fold/CSE.
func.func @p02(%A: memref<4xi32>, %a: i32, %b: i32, %c: i32, %d: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %ab = arith.muli %a, %b : i32
  %ac = arith.muli %a, %c : i32
  %ad = arith.muli %a, %d : i32
  %bc = arith.muli %b, %c : i32
  memref.store %ab, %A[%c0] : memref<4xi32>
  memref.store %ac, %A[%c1] : memref<4xi32>
  memref.store %ad, %A[%c2] : memref<4xi32>
  memref.store %bc, %A[%c3] : memref<4xi32>
  return
}
