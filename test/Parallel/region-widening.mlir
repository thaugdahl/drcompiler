// RUN: dr-opt %s --pass-pipeline='builtin.module(dr-par-bubbles{par-test-regions})' -verify-diagnostics

// M1: widen seeded bubbles to a fixed point and report MAXIMAL regions.
//  - climb collapses a perfect affine band into one region (per-level par/seq).
//  - fuse merges adjacent conformant sibling bands over provably-disjoint data.
//  - a rejected fuse is a frozen frontier with a reason.

// ---- climb: GEMM perfect band => i,j parallel, k carried (one region) -------
func.func @gemm() {
  %A = memref.alloc() : memref<64x64xf32>
  %B = memref.alloc() : memref<64x64xf32>
  %C = memref.alloc() : memref<64x64xf32>
  // expected-remark @below {{par-region: bands=1 par=[0,1] seq=[2]}}
  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 64 {
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

// ---- fuse: conformant siblings on DISJOINT allocs => one fused region --------
func.func @fuse_disjoint() {
  %A = memref.alloc() : memref<128xf32>
  %T = memref.alloc() : memref<128xf32>
  %C = memref.alloc() : memref<128xf32>
  %z = arith.constant 0.0 : f32
  // expected-remark @below {{par-region: bands=2 par=[0] seq=[]}}
  affine.for %i = 0 to 128 {
    affine.store %z, %C[%i] : memref<128xf32>
  }
  affine.for %i = 0 to 128 {
    %a = affine.load %A[%i] : memref<128xf32>
    %e = math.exp %a : f32
    affine.store %e, %T[%i] : memref<128xf32>
  }
  return
}

// ---- reject: both touch the SAME buffer (write) => frozen (shared-write) -----
func.func @no_fuse_shared() {
  %C = memref.alloc() : memref<128xf32>
  %z = arith.constant 0.0 : f32
  %c1 = arith.constant 1.0 : f32
  // expected-remark @below {{par-region: bands=1 par=[0] seq=[]}}
  affine.for %i = 0 to 128 {
    affine.store %z, %C[%i] : memref<128xf32>
  }
  // expected-remark @below {{par-frontier: frozen (shared-write)}}
  // expected-remark @below {{par-region: bands=1 par=[0] seq=[]}}
  affine.for %i = 0 to 128 {
    %c = affine.load %C[%i] : memref<128xf32>
    %s = arith.addf %c, %c1 : f32
    affine.store %s, %C[%i] : memref<128xf32>
  }
  return
}

// ---- reject: different bounds => frozen (non-conformant) --------------------
func.func @no_fuse_bounds() {
  %A = memref.alloc() : memref<128xf32>
  %B = memref.alloc() : memref<64xf32>
  %z = arith.constant 0.0 : f32
  // expected-remark @below {{par-region: bands=1 par=[0] seq=[]}}
  affine.for %i = 0 to 128 {
    affine.store %z, %A[%i] : memref<128xf32>
  }
  // expected-remark @below {{par-frontier: frozen (non-conformant)}}
  // expected-remark @below {{par-region: bands=1 par=[0] seq=[]}}
  affine.for %i = 0 to 64 {
    affine.store %z, %B[%i] : memref<64xf32>
  }
  return
}

// ---- reject: predecessor axis is sequential => frozen (outer-sequential) ----
func.func @no_fuse_seq() {
  %A = memref.alloc() : memref<128xf32>
  %B = memref.alloc() : memref<128xf32>
  %c1 = arith.constant 1.0 : f32
  %z = arith.constant 0.0 : f32
  // expected-remark @below {{par-region: bands=1 par=[] seq=[0]}}
  affine.for %i = 1 to 128 {
    %p = affine.load %A[%i - 1] : memref<128xf32>
    %s = arith.addf %p, %c1 : f32
    affine.store %s, %A[%i] : memref<128xf32>
  }
  // expected-remark @below {{par-frontier: frozen (outer-sequential)}}
  // expected-remark @below {{par-region: bands=1 par=[0] seq=[]}}
  affine.for %i = 0 to 128 {
    affine.store %z, %B[%i] : memref<128xf32>
  }
  return
}
