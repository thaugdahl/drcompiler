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
    %buf = memref.alloc() : memref<524288xf64>
    %result = affine.for %rep = 0 to 50 iter_args(%outer = %cst) -> (f64) {
      affine.for %i = 0 to 524288 {
        %ci = affine.apply affine_map<(d0) -> (d0 mod 16)>(%i)
        %cv = affine.load %lut[%ci] : memref<16xf64>
        %m = arith.mulf %cv, %c2 : f64
        %val = arith.addf %m, %c1 : f64
        affine.store %val, %buf[%i] : memref<524288xf64>
      }
      %inner = affine.for %j = 0 to 524288 iter_args(%acc = %cst) -> (f64) {
        %v = affine.load %buf[%j] : memref<524288xf64>
        %s = arith.addf %acc, %v : f64
        affine.yield %s : f64
      }
      %next = arith.addf %outer, %inner : f64
      affine.yield %next : f64
    }
    memref.dealloc %lut : memref<16xf64>
    memref.dealloc %buf : memref<524288xf64>
    %ret = arith.fptosi %result : f64 to i32
    return %ret : i32
  }
}
