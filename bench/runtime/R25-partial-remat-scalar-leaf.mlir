// DR_PIPELINE: builtin.module(data-recomputation{dr-recompute=true dr-cost-model=true dr-partial-remat=true})
module {
  func.func @main() -> i32 {
    %cst = arith.constant 0.0 : f64
    %c42 = arith.constant 42.0 : f64
    %c2 = arith.constant 2.0 : f64
    %scalar = memref.alloc() : memref<f64>
    memref.store %c42, %scalar[] : memref<f64>
    %buf = memref.alloc() : memref<524288xf64>
    %result = affine.for %rep = 0 to 50 iter_args(%outer = %cst) -> (f64) {
      affine.for %i = 0 to 524288 {
        %sv = memref.load %scalar[] : memref<f64>
        %idx = arith.index_cast %i : index to i32
        %f = arith.sitofp %idx : i32 to f64
        %m = arith.mulf %f, %sv : f64
        %val = arith.addf %m, %c2 : f64
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
    memref.dealloc %scalar : memref<f64>
    memref.dealloc %buf : memref<524288xf64>
    %ret = arith.fptosi %result : f64 to i32
    return %ret : i32
  }
}
