// DR_PIPELINE: builtin.module(data-recomputation{dr-recompute=true dr-cost-model=true dr-partial-remat=true})
module {
  memref.global "private" constant @weights : memref<16xf64> = dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]>
  func.func @main() -> i32 {
    %cst = arith.constant 0.0 : f64
    %c2 = arith.constant 2.0 : f64
    %wt = memref.get_global @weights : memref<16xf64>
    %buf = memref.alloc() : memref<2097152xf64>
    %result = affine.for %rep = 0 to 30 iter_args(%outer = %cst) -> (f64) {
      affine.for %i = 0 to 2097152 {
        %ci = affine.apply affine_map<(d0) -> (d0 mod 16)>(%i)
        %cv = affine.load %wt[%ci] : memref<16xf64>
        %val = arith.mulf %cv, %c2 : f64
        affine.store %val, %buf[%i] : memref<2097152xf64>
      }
      %inner = affine.for %j = 0 to 2097152 iter_args(%acc = %cst) -> (f64) {
        %v = affine.load %buf[%j] : memref<2097152xf64>
        %s = arith.addf %acc, %v : f64
        affine.yield %s : f64
      }
      %next = arith.addf %outer, %inner : f64
      affine.yield %next : f64
    }
    memref.dealloc %buf : memref<2097152xf64>
    %ret = arith.fptosi %result : f64 to i32
    return %ret : i32
  }
}
