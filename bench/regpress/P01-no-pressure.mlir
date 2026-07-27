// Baseline: 2 args, no pressure, no spills expected.
func.func @p01(%A: memref<1xi32>, %a: i32, %b: i32) {
  %c0 = arith.constant 0 : index
  %r = arith.addi %a, %b : i32
  memref.store %r, %A[%c0] : memref<1xi32>
  return
}
