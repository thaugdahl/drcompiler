// RUN: dr-opt %s --pass-pipeline='builtin.module(dr-par-bubbles{par-test-diagnostics})' -verify-diagnostics

// M0 of the bubble-widening track: seed one bubble per loop and classify its
// axis via the ParAliasOracle.  Diagnostic-only (no IR mutation).

// ---- init: single write, no other access to B  => PARALLEL -----------------
func.func @init() {
  %z = arith.constant 0.0 : f32
  %B = memref.alloc() : memref<128xf32>
  // expected-remark @below {{par-bubble axis: PARALLEL}}
  affine.for %i = 0 to 128 {
    affine.store %z, %B[%i] : memref<128xf32>
  }
  return
}

// ---- pointwise across two DISTINCT allocs (Tier 0 separates) => PARALLEL ----
func.func @pointwise() {
  %A = memref.alloc() : memref<128xf32>
  %B = memref.alloc() : memref<128xf32>
  // expected-remark @below {{par-bubble axis: PARALLEL}}
  affine.for %i = 0 to 128 {
    %a = affine.load %A[%i] : memref<128xf32>
    %e = math.exp %a : f32
    affine.store %e, %B[%i] : memref<128xf32>
  }
  return
}

// ---- offset-1 read/write on the SAME alloc (Tier 1 exact) => carried --------
func.func @carried() {
  %A = memref.alloc() : memref<128xf32>
  %c1 = arith.constant 1.0 : f32
  // expected-remark @below {{par-bubble axis: SEQUENTIAL (carried dependence)}}
  affine.for %i = 1 to 128 {
    %p = affine.load %A[%i - 1] : memref<128xf32>
    %s = arith.addf %p, %c1 : f32
    affine.store %s, %A[%i] : memref<128xf32>
  }
  return
}

// ---- iter_args accumulator => reduction (sequential in v1) ------------------
func.func @reduction() -> f32 {
  %A = memref.alloc() : memref<128xf32>
  %z = arith.constant 0.0 : f32
  // expected-remark @below {{par-bubble axis: SEQUENTIAL (reduction)}}
  %r = affine.for %i = 0 to 128 iter_args(%acc = %z) -> f32 {
    %a = affine.load %A[%i] : memref<128xf32>
    %s = arith.addf %acc, %a : f32
    affine.yield %s : f32
  }
  return %r : f32
}

// ---- opaque call in the body => conservative --------------------------------
func.func private @sink(f32)
func.func @opaque() {
  %A = memref.alloc() : memref<128xf32>
  // expected-remark @below {{par-bubble axis: SEQUENTIAL (conservative)}}
  affine.for %i = 0 to 128 {
    %a = affine.load %A[%i] : memref<128xf32>
    func.call @sink(%a) : (f32) -> ()
  }
  return
}

// ---- distinct function-arg memrefs: non-aliasing (affine model) => PARALLEL --
// Distinct allocation roots (here, distinct func args) are assumed disjoint,
// consistent with affine::isLoopParallel / checkMemrefAccessDependence.
func.func @args(%A: memref<128xf32>, %B: memref<128xf32>) {
  // expected-remark @below {{par-bubble axis: PARALLEL}}
  affine.for %i = 0 to 128 {
    %a = affine.load %A[%i] : memref<128xf32>
    affine.store %a, %B[%i] : memref<128xf32>
  }
  return
}

// ---- two views of the SAME buffer may overlap => conservative (safer than
// raw affine, which would treat distinct SSA memrefs as independent).
func.func @same_root_views(%Buf: memref<256xf32>) {
  %lo = memref.subview %Buf[0]  [128] [1]
      : memref<256xf32> to memref<128xf32, strided<[1]>>
  %hi = memref.subview %Buf[64] [128] [1]
      : memref<256xf32> to memref<128xf32, strided<[1], offset: 64>>
  // expected-remark @below {{par-bubble axis: SEQUENTIAL (conservative)}}
  affine.for %i = 0 to 128 {
    %v = affine.load %lo[%i] : memref<128xf32, strided<[1]>>
    affine.store %v, %hi[%i] : memref<128xf32, strided<[1], offset: 64>>
  }
  return
}

// ---- GEMM: i,j parallel; k carries the C[i,j] reduction => k sequential -----
func.func @gemm() {
  %A = memref.alloc() : memref<64x64xf32>
  %B = memref.alloc() : memref<64x64xf32>
  %C = memref.alloc() : memref<64x64xf32>
  // expected-remark @below {{par-bubble axis: PARALLEL}}
  affine.for %i = 0 to 64 {
    // expected-remark @below {{par-bubble axis: PARALLEL}}
    affine.for %j = 0 to 64 {
      // expected-remark @below {{par-bubble axis: SEQUENTIAL (carried dependence)}}
      affine.for %k = 0 to 64 {
        %a = affine.load %A[%i, %k] : memref<64x64xf32>
        %b = affine.load %B[%k, %j] : memref<64x64xf32>
        %c = affine.load %C[%i, %j] : memref<64x64xf32>
        %p = arith.mulf %a, %b : f32
        %s = arith.addf %c, %p : f32
        affine.store %s, %C[%i, %j] : memref<64x64xf32>
      }
    }
  }
  return
}
