// DR_PIPELINE: builtin.module(data-recomputation{dr-recompute=true dr-cost-model=true dr-partial-remat=true})
module {
  func.func @main() -> i32 {
    %cst = arith.constant 0.0 : f64
    %c2 = arith.constant 2.0 : f64
    %c1 = arith.constant 1.0 : f64
    %lut = memref.alloc() : memref<16xf64>
    affine.for %w = 0 to 16 {
      %wi = arith.index_cast %w : index to i32
      %wf = arith.sitofp %wi : i32 to f64
      %wv = arith.addf %wf, %c1 : f64
      affine.store %wv, %lut[%w] : memref<16xf64>
    }
    %buf = memref.alloc() : memref<2097152xf64>
    %result = affine.for %rep = 0 to 20 iter_args(%outer = %cst) -> (f64) {
      affine.for %i = 0 to 2097152 {
        %ci = affine.apply affine_map<(d0) -> (d0 mod 16)>(%i)
        %cv = affine.load %lut[%ci] : memref<16xf64>
        %val = arith.mulf %cv, %c2 : f64
        affine.store %val, %buf[%i] : memref<2097152xf64>
      }
      %s1 = affine.for %j = 0 to 2097152 iter_args(%acc = %cst) -> (f64) {
        %v = affine.load %buf[%j] : memref<2097152xf64>
        %s = arith.addf %acc, %v : f64
        affine.yield %s : f64
      }
      %s2 = affine.for %j = 0 to 2097152 iter_args(%best = %cst) -> (f64) {
        %v = affine.load %buf[%j] : memref<2097152xf64>
        %gt = arith.cmpf ogt, %v, %best : f64
        %sel = arith.select %gt, %v, %best : f64
        affine.yield %sel : f64
      }
      %combined = arith.addf %s1, %s2 : f64
      %next = arith.addf %outer, %combined : f64
      affine.yield %next : f64
    }
    memref.dealloc %lut : memref<16xf64>
    memref.dealloc %buf : memref<2097152xf64>
    %ret = arith.fptosi %result : f64 to i32
    return %ret : i32
  }
}
