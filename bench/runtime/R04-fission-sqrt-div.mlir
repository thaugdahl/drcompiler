// DR_PIPELINE: builtin.module(memory-fission)
//
// R04: Two sibling loops with identical sqrt+div chain from same source.
// Fission should hoist shared computation into producer loop + buffer.
// Expected: fission faster (compute once, load twice vs compute twice).

module {
  func.func @main() -> i32 {
    %cst0 = arith.constant 0.0 : f64
    %cst1 = arith.constant 1.0 : f64
    %eps = arith.constant 0.001 : f64

    %src = memref.alloc() : memref<100000xf64>
    affine.for %i = 0 to 100000 {
      %idx = arith.index_cast %i : index to i32
      %f = arith.sitofp %idx : i32 to f64
      %val = arith.addf %f, %cst1 : f64
      affine.store %val, %src[%i] : memref<100000xf64>
    }

    %result = affine.for %rep = 0 to 200 iter_args(%outer = %cst0) -> (f64) {
      // Loop 1: sum of sqrt(x^2+1)/(x+eps)
      %sum = affine.for %i = 0 to 100000 iter_args(%acc = %cst0) -> (f64) {
        %xi = affine.load %src[%i] : memref<100000xf64>
        %sq = arith.mulf %xi, %xi : f64
        %s = arith.addf %sq, %cst1 : f64
        %r = math.sqrt %s : f64
        %d = arith.addf %xi, %eps : f64
        %v = arith.divf %r, %d : f64
        %out = arith.addf %acc, %v : f64
        affine.yield %out : f64
      }

      // Loop 2: max of sqrt(x^2+1)/(x+eps) — identical computation.
      %max = affine.for %i = 0 to 100000 iter_args(%best = %cst0) -> (f64) {
        %xi = affine.load %src[%i] : memref<100000xf64>
        %sq = arith.mulf %xi, %xi : f64
        %s = arith.addf %sq, %cst1 : f64
        %r = math.sqrt %s : f64
        %d = arith.addf %xi, %eps : f64
        %v = arith.divf %r, %d : f64
        %gt = arith.cmpf ogt, %v, %best : f64
        %sel = arith.select %gt, %v, %best : f64
        affine.yield %sel : f64
      }

      %combined = arith.addf %sum, %max : f64
      %next = arith.addf %outer, %combined : f64
      affine.yield %next : f64
    }

    memref.dealloc %src : memref<100000xf64>
    %ret = arith.fptosi %result : f64 to i32
    return %ret : i32
  }
}
