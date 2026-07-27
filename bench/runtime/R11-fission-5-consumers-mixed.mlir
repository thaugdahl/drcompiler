// DR_PIPELINE: builtin.module(memory-fission)
// R11: 5 sibling loops with identical exp(x)*log(x+1) chain.
// Each does a different reduction. Fission: 5→1 compute + 5 loads.

module {
  func.func @main() -> i32 {
    %cst0 = arith.constant 0.0 : f64
    %cst1 = arith.constant 1.0 : f64
    %large = arith.constant 1.0e30 : f64

    %src = memref.alloc() : memref<40000xf64>
    affine.for %i = 0 to 40000 {
      %idx = arith.index_cast %i : index to i32
      %f = arith.sitofp %idx : i32 to f64
      %scaled = arith.divf %f, %large : f64
      %val = arith.addf %scaled, %cst1 : f64
      affine.store %val, %src[%i] : memref<40000xf64>
    }

    %result = affine.for %rep = 0 to 300 iter_args(%outer = %cst0) -> (f64) {
      // Loop 1: sum
      %s1 = affine.for %i = 0 to 40000 iter_args(%acc = %cst0) -> (f64) {
        %xi = affine.load %src[%i] : memref<40000xf64>
        %lx = math.log %xi : f64
        %ex = math.exp %lx : f64
        %out = arith.addf %acc, %ex : f64
        affine.yield %out : f64
      }
      // Loop 2: max
      %s2 = affine.for %i = 0 to 40000 iter_args(%best = %cst0) -> (f64) {
        %xi = affine.load %src[%i] : memref<40000xf64>
        %lx = math.log %xi : f64
        %ex = math.exp %lx : f64
        %gt = arith.cmpf ogt, %ex, %best : f64
        %sel = arith.select %gt, %ex, %best : f64
        affine.yield %sel : f64
      }
      // Loop 3: sum of squares
      %s3 = affine.for %i = 0 to 40000 iter_args(%acc = %cst0) -> (f64) {
        %xi = affine.load %src[%i] : memref<40000xf64>
        %lx = math.log %xi : f64
        %ex = math.exp %lx : f64
        %sq = arith.mulf %ex, %ex : f64
        %out = arith.addf %acc, %sq : f64
        affine.yield %out : f64
      }
      // Loop 4: negative sum
      %s4 = affine.for %i = 0 to 40000 iter_args(%acc = %cst0) -> (f64) {
        %xi = affine.load %src[%i] : memref<40000xf64>
        %lx = math.log %xi : f64
        %ex = math.exp %lx : f64
        %neg = arith.negf %ex : f64
        %out = arith.addf %acc, %neg : f64
        affine.yield %out : f64
      }
      // Loop 5: reciprocal sum
      %s5 = affine.for %i = 0 to 40000 iter_args(%acc = %cst0) -> (f64) {
        %xi = affine.load %src[%i] : memref<40000xf64>
        %lx = math.log %xi : f64
        %ex = math.exp %lx : f64
        %recip = arith.divf %cst1, %ex : f64
        %out = arith.addf %acc, %recip : f64
        affine.yield %out : f64
      }
      %t1 = arith.addf %s1, %s2 : f64
      %t2 = arith.addf %s3, %s4 : f64
      %t3 = arith.addf %t1, %t2 : f64
      %combined = arith.addf %t3, %s5 : f64
      %next = arith.addf %outer, %combined : f64
      affine.yield %next : f64
    }
    memref.dealloc %src : memref<40000xf64>
    %ret = arith.fptosi %result : f64 to i32
    return %ret : i32
  }
}
