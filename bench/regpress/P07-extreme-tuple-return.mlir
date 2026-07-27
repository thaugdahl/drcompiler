// 20-result function: all values must be live at the return.  This forces
// register pressure since SysV calling convention can only return small
// aggregates in 2 regs.
func.func @p07(%a: i32, %b: i32, %c: i32, %d: i32, %e: i32)
    -> (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
        i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) {
  %v00 = arith.muli %a, %b : i32
  %v01 = arith.muli %a, %c : i32
  %v02 = arith.muli %a, %d : i32
  %v03 = arith.muli %a, %e : i32
  %v04 = arith.muli %b, %c : i32
  %v05 = arith.muli %b, %d : i32
  %v06 = arith.muli %b, %e : i32
  %v07 = arith.muli %c, %d : i32
  %v08 = arith.muli %c, %e : i32
  %v09 = arith.muli %d, %e : i32
  %v10 = arith.addi %v00, %v01 : i32
  %v11 = arith.addi %v00, %v02 : i32
  %v12 = arith.addi %v00, %v03 : i32
  %v13 = arith.addi %v00, %v04 : i32
  %v14 = arith.addi %v01, %v05 : i32
  %v15 = arith.addi %v01, %v06 : i32
  %v16 = arith.addi %v02, %v07 : i32
  %v17 = arith.addi %v03, %v08 : i32
  %v18 = arith.addi %v04, %v09 : i32
  %v19 = arith.addi %v05, %v09 : i32
  return %v00, %v01, %v02, %v03, %v04, %v05, %v06, %v07, %v08, %v09,
         %v10, %v11, %v12, %v13, %v14, %v15, %v16, %v17, %v18, %v19
    : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
      i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
}
