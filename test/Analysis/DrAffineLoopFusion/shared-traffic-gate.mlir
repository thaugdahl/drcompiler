// RUN: dr-opt %s --pass-pipeline='builtin.module(dr-affine-loop-fusion{emit-rationale=true})' -verify-diagnostics

// Shared-traffic gate: fusion only removes cost on data BOTH nests touch.
// Fusing a tiny init into a nest that streams an unrelated large array
// passes the cycle-total comparison (smaller abstract total) but saves
// almost no traffic and risks breaking backend vectorization of the big
// nest — measured 5x slower on atax at EXTRALARGE.  The gate requires the
// overlap to be >= 5% of the combined footprint.

func.func @tiny_overlap(%y: memref<64xf64>, %A: memref<2048x2048xf64>, %out: memref<2048xf64>) {
  %cst = arith.constant 0.0 : f64
  // expected-remark @below {{fusion-rationale: REJECT reason=no-shared-traffic shared=8 combined=33571336}}
  affine.for %j = 0 to 64 {
    affine.store %cst, %y[%j] : memref<64xf64>
  }
  affine.for %i = 0 to 2048 {
    affine.for %j = 0 to 2048 {
      %a = affine.load %A[%i, %j] : memref<2048x2048xf64>
      %o = affine.load %out[%i] : memref<2048xf64>
      %s = arith.addf %o, %a : f64
      affine.store %s, %out[%i] : memref<2048xf64>
    }
    %yv = affine.load %y[0] : memref<64xf64>
    %ov = affine.load %out[%i] : memref<2048xf64>
    %f = arith.addf %ov, %yv : f64
    affine.store %f, %out[%i] : memref<2048xf64>
  }
  return
}

// Producer fully consumed by the consumer (the whole intermediate array is
// shared): the gate must let this through — forwarding T's stores to T's
// loads is the textbook fusion win.

func.func @full_overlap(%A: memref<256x256xf64>, %B: memref<256x256xf64>, %C: memref<256x256xf64>) {
  // expected-remark @below {{fusion-rationale: FUSE}}
  affine.for %i = 0 to 256 {
    affine.for %j = 0 to 256 {
      %a = affine.load %A[%i, %j] : memref<256x256xf64>
      %two = arith.constant 2.0 : f64
      %m = arith.mulf %a, %two : f64
      affine.store %m, %B[%i, %j] : memref<256x256xf64>
    }
  }
  affine.for %i = 0 to 256 {
    affine.for %j = 0 to 256 {
      %b = affine.load %B[%i, %j] : memref<256x256xf64>
      %one = arith.constant 1.0 : f64
      %s = arith.addf %b, %one : f64
      affine.store %s, %C[%i, %j] : memref<256x256xf64>
    }
  }
  return
}
