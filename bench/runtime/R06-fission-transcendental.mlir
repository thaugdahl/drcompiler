// DR_PIPELINE: builtin.module(memory-fission)
//
// R06: Three sibling loops with exp+log chain — very expensive (40cy each).
// Fission: compute once (40cy) + 3 loads (4cy each) vs recompute 3x (120cy).
// Expected: fission much faster.

module {
  func.func @main() -> i32 {
    %cst0 = arith.constant 0.0 : f64
    %cst1 = arith.constant 1.0 : f64

    %src = memref.alloc() : memref<50000xf64>
    affine.for %i = 0 to 50000 {
      %idx = arith.index_cast %i : index to i32
      %f = arith.sitofp %idx : i32 to f64
      %val = arith.addf %f, %cst1 : f64
      affine.store %val, %src[%i] : memref<50000xf64>
    }

    %result = affine.for %rep = 0 to 200 iter_args(%outer = %cst0) -> (f64) {
      // Loop 1
      %s1 = affine.for %i = 0 to 50000 iter_args(%acc = %cst0) -> (f64) {
        %xi = affine.load %src[%i] : memref<50000xf64>
        %e = math.exp %xi : f64
        %v = math.log %e : f64
        %out = arith.addf %acc, %v : f64
        affine.yield %out : f64
      }
      // Loop 2 — identical chain
      %s2 = affine.for %i = 0 to 50000 iter_args(%acc = %cst0) -> (f64) {
        %xi = affine.load %src[%i] : memref<50000xf64>
        %e = math.exp %xi : f64
        %v = math.log %e : f64
        %out = arith.addf %acc, %v : f64
        affine.yield %out : f64
      }
      // Loop 3 — identical chain
      %s3 = affine.for %i = 0 to 50000 iter_args(%acc = %cst0) -> (f64) {
        %xi = affine.load %src[%i] : memref<50000xf64>
        %e = math.exp %xi : f64
        %v = math.log %e : f64
        %out = arith.addf %acc, %v : f64
        affine.yield %out : f64
      }
      %t = arith.addf %s1, %s2 : f64
      %combined = arith.addf %t, %s3 : f64
      %next = arith.addf %outer, %combined : f64
      affine.yield %next : f64
    }

    memref.dealloc %src : memref<50000xf64>
    %ret = arith.fptosi %result : f64 to i32
    return %ret : i32
  }
}
