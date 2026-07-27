// DR_PIPELINE: builtin.module(memory-fission)
// R10: 4 sibling loops with identical sqrt(x^2+1)/(x+eps) chain.
// Fission: 4 computes → 1 compute + 4 loads. ~3-4x expected.

module {
  func.func @main() -> i32 {
    %cst0 = arith.constant 0.0 : f64
    %cst1 = arith.constant 1.0 : f64
    %eps = arith.constant 0.001 : f64

    %src = memref.alloc() : memref<50000xf64>
    affine.for %i = 0 to 50000 {
      %idx = arith.index_cast %i : index to i32
      %f = arith.sitofp %idx : i32 to f64
      %val = arith.addf %f, %cst1 : f64
      affine.store %val, %src[%i] : memref<50000xf64>
    }

    %result = affine.for %rep = 0 to 500 iter_args(%outer = %cst0) -> (f64) {
      %s1 = affine.for %i = 0 to 50000 iter_args(%acc = %cst0) -> (f64) {
        %xi = affine.load %src[%i] : memref<50000xf64>
        %sq = arith.mulf %xi, %xi : f64
        %s = arith.addf %sq, %cst1 : f64
        %r = math.sqrt %s : f64
        %d = arith.addf %xi, %eps : f64
        %v = arith.divf %r, %d : f64
        %out = arith.addf %acc, %v : f64
        affine.yield %out : f64
      }
      %s2 = affine.for %i = 0 to 50000 iter_args(%best = %cst0) -> (f64) {
        %xi = affine.load %src[%i] : memref<50000xf64>
        %sq = arith.mulf %xi, %xi : f64
        %s = arith.addf %sq, %cst1 : f64
        %r = math.sqrt %s : f64
        %d = arith.addf %xi, %eps : f64
        %v = arith.divf %r, %d : f64
        %gt = arith.cmpf ogt, %v, %best : f64
        %sel = arith.select %gt, %v, %best : f64
        affine.yield %sel : f64
      }
      %s3 = affine.for %i = 0 to 50000 iter_args(%acc = %cst0) -> (f64) {
        %xi = affine.load %src[%i] : memref<50000xf64>
        %sq = arith.mulf %xi, %xi : f64
        %s = arith.addf %sq, %cst1 : f64
        %r = math.sqrt %s : f64
        %d = arith.addf %xi, %eps : f64
        %v = arith.divf %r, %d : f64
        %neg = arith.negf %v : f64
        %out = arith.addf %acc, %neg : f64
        affine.yield %out : f64
      }
      %s4 = affine.for %i = 0 to 50000 iter_args(%acc = %cst0) -> (f64) {
        %xi = affine.load %src[%i] : memref<50000xf64>
        %sq = arith.mulf %xi, %xi : f64
        %s = arith.addf %sq, %cst1 : f64
        %r = math.sqrt %s : f64
        %d = arith.addf %xi, %eps : f64
        %v = arith.divf %r, %d : f64
        %sq2 = arith.mulf %v, %v : f64
        %out = arith.addf %acc, %sq2 : f64
        affine.yield %out : f64
      }
      %t1 = arith.addf %s1, %s2 : f64
      %t2 = arith.addf %s3, %s4 : f64
      %combined = arith.addf %t1, %t2 : f64
      %next = arith.addf %outer, %combined : f64
      affine.yield %next : f64
    }
    memref.dealloc %src : memref<50000xf64>
    %ret = arith.fptosi %result : f64 to i32
    return %ret : i32
  }
}
