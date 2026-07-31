// Falcon-as-oracle spike corpus.
//
// Static-shape affine nests at PolyBench LARGE sizes, chosen to exercise the
// three places drcompiler::reuse::analyzeBandReuse is known to approximate:
//
//   gemm, 2mm-inner   — control: distinct arrays, exact trip counts.  Our
//                       footprint should already be right here.
//   jacobi-2d, heat-3d,
//   seidel-2d         — stencil groups: several references to ONE array that
//                       differ only in constant offset.  footprintBytes sums
//                       them instead of taking their union.
//   syrk, trmm,
//   covariance        — triangular bounds: no constant trip count, so the
//                       analysis either fails or (with the tiler's
//                       accept-trip-upper-bounds) uses the rectangular
//                       bounding count, ~2x the real iteration space.
//
// The reference kernels use the PolyBench space nest for one time step; the
// time loop is dropped so every function is a single top-level perfect band
// (what dr-test-reuse-analysis walks).

// ---------------------------------------------------------------- controls --

// gemm, LARGE: NI=1000 NJ=1100 NK=1200.
// True bytes touched: C 1000*1100*8 = 8.80 MB, A 1000*1200*8 = 9.60 MB,
// B 1200*1100*8 = 10.56 MB  =>  28.96 MB.
func.func @gemm(%C: memref<1000x1100xf64>, %A: memref<1000x1200xf64>,
                %B: memref<1200x1100xf64>, %alpha: f64) {
  affine.for %i = 0 to 1000 {
    affine.for %k = 0 to 1200 {
      affine.for %j = 0 to 1100 {
        %a = affine.load %A[%i, %k] : memref<1000x1200xf64>
        %b = affine.load %B[%k, %j] : memref<1200x1100xf64>
        %c = affine.load %C[%i, %j] : memref<1000x1100xf64>
        %t = arith.mulf %alpha, %a : f64
        %m = arith.mulf %t, %b : f64
        %s = arith.addf %c, %m : f64
        affine.store %s, %C[%i, %j] : memref<1000x1100xf64>
      }
    }
  }
  return
}

// 3mm first product, LARGE: NI=800 NJ=900 NK=1000.  Second control.
func.func @mm_800_900_1000(%E: memref<800x900xf64>, %A: memref<800x1000xf64>,
                           %B: memref<1000x900xf64>) {
  affine.for %i = 0 to 800 {
    affine.for %j = 0 to 900 {
      affine.for %k = 0 to 1000 {
        %a = affine.load %A[%i, %k] : memref<800x1000xf64>
        %b = affine.load %B[%k, %j] : memref<1000x900xf64>
        %e = affine.load %E[%i, %j] : memref<800x900xf64>
        %m = arith.mulf %a, %b : f64
        %s = arith.addf %e, %m : f64
        affine.store %s, %E[%i, %j] : memref<800x900xf64>
      }
    }
  }
  return
}

// -------------------------------------------------------- stencil groups ----

// jacobi-2d space step, LARGE: N=1300.
// True bytes: A interior+halo 1300*1300*8 = 13.52 MB, B 1298*1298*8 = 13.48 MB
// => 27.00 MB.  Five A references, one B store.
func.func @jacobi2d(%A: memref<1300x1300xf64>, %B: memref<1300x1300xf64>) {
  %c = arith.constant 2.000000e-01 : f64
  affine.for %i = 1 to 1299 {
    affine.for %j = 1 to 1299 {
      %0 = affine.load %A[%i, %j] : memref<1300x1300xf64>
      %1 = affine.load %A[%i, %j - 1] : memref<1300x1300xf64>
      %2 = affine.load %A[%i, %j + 1] : memref<1300x1300xf64>
      %3 = affine.load %A[%i + 1, %j] : memref<1300x1300xf64>
      %4 = affine.load %A[%i - 1, %j] : memref<1300x1300xf64>
      %s1 = arith.addf %0, %1 : f64
      %s2 = arith.addf %s1, %2 : f64
      %s3 = arith.addf %s2, %3 : f64
      %s4 = arith.addf %s3, %4 : f64
      %r = arith.mulf %s4, %c : f64
      affine.store %r, %B[%i, %j] : memref<1300x1300xf64>
    }
  }
  return
}

// heat-3d space step, LARGE: N=120.
// True bytes: A 120^3*8 = 13.82 MB, B 118^3*8 = 13.14 MB => 26.96 MB.
func.func @heat3d(%A: memref<120x120x120xf64>, %B: memref<120x120x120xf64>) {
  %c = arith.constant 1.250000e-01 : f64
  affine.for %i = 1 to 119 {
    affine.for %j = 1 to 119 {
      affine.for %k = 1 to 119 {
        %0 = affine.load %A[%i, %j, %k] : memref<120x120x120xf64>
        %1 = affine.load %A[%i - 1, %j, %k] : memref<120x120x120xf64>
        %2 = affine.load %A[%i + 1, %j, %k] : memref<120x120x120xf64>
        %3 = affine.load %A[%i, %j - 1, %k] : memref<120x120x120xf64>
        %4 = affine.load %A[%i, %j + 1, %k] : memref<120x120x120xf64>
        %5 = affine.load %A[%i, %j, %k - 1] : memref<120x120x120xf64>
        %6 = affine.load %A[%i, %j, %k + 1] : memref<120x120x120xf64>
        %s1 = arith.addf %0, %1 : f64
        %s2 = arith.addf %s1, %2 : f64
        %s3 = arith.addf %s2, %3 : f64
        %s4 = arith.addf %s3, %4 : f64
        %s5 = arith.addf %s4, %5 : f64
        %s6 = arith.addf %s5, %6 : f64
        %r = arith.mulf %s6, %c : f64
        affine.store %r, %B[%i, %j, %k] : memref<120x120x120xf64>
      }
    }
  }
  return
}

// seidel-2d space step, LARGE: N=1200.  In-place 9-point: nine references to
// ONE array, no second buffer.  True bytes: 1200*1200*8 = 11.52 MB.
func.func @seidel2d(%A: memref<1200x1200xf64>) {
  %c9 = arith.constant 9.000000e+00 : f64
  affine.for %i = 1 to 1199 {
    affine.for %j = 1 to 1199 {
      %0 = affine.load %A[%i - 1, %j - 1] : memref<1200x1200xf64>
      %1 = affine.load %A[%i - 1, %j] : memref<1200x1200xf64>
      %2 = affine.load %A[%i - 1, %j + 1] : memref<1200x1200xf64>
      %3 = affine.load %A[%i, %j - 1] : memref<1200x1200xf64>
      %4 = affine.load %A[%i, %j] : memref<1200x1200xf64>
      %5 = affine.load %A[%i, %j + 1] : memref<1200x1200xf64>
      %6 = affine.load %A[%i + 1, %j - 1] : memref<1200x1200xf64>
      %7 = affine.load %A[%i + 1, %j] : memref<1200x1200xf64>
      %8 = affine.load %A[%i + 1, %j + 1] : memref<1200x1200xf64>
      %s1 = arith.addf %0, %1 : f64
      %s2 = arith.addf %s1, %2 : f64
      %s3 = arith.addf %s2, %3 : f64
      %s4 = arith.addf %s3, %4 : f64
      %s5 = arith.addf %s4, %5 : f64
      %s6 = arith.addf %s5, %6 : f64
      %s7 = arith.addf %s6, %7 : f64
      %s8 = arith.addf %s7, %8 : f64
      %r = arith.divf %s8, %c9 : f64
      affine.store %r, %A[%i, %j] : memref<1200x1200xf64>
    }
  }
  return
}

// ---------------------------------------------------- triangular bounds -----

// syrk compute nest, LARGE: N=1200 M=1000.  j runs 0..i, so the real
// iteration space is N*(N+1)/2*M = 720.6M points, not N*N*M = 1440M.
func.func @syrk(%C: memref<1200x1200xf64>, %A: memref<1200x1000xf64>,
                %alpha: f64) {
  affine.for %i = 0 to 1200 {
    affine.for %k = 0 to 1000 {
      affine.for %j = 0 to affine_map<(d0) -> (d0 + 1)>(%i) {
        %aik = affine.load %A[%i, %k] : memref<1200x1000xf64>
        %ajk = affine.load %A[%j, %k] : memref<1200x1000xf64>
        %c = affine.load %C[%i, %j] : memref<1200x1200xf64>
        %t = arith.mulf %alpha, %aik : f64
        %m = arith.mulf %t, %ajk : f64
        %s = arith.addf %c, %m : f64
        affine.store %s, %C[%i, %j] : memref<1200x1200xf64>
      }
    }
  }
  return
}

// trmm compute nest, LARGE: M=1000 N=1200.  k runs i+1..M.
func.func @trmm(%A: memref<1000x1000xf64>, %B: memref<1000x1200xf64>) {
  affine.for %i = 0 to 1000 {
    affine.for %j = 0 to 1200 {
      affine.for %k = affine_map<(d0) -> (d0 + 1)>(%i) to 1000 {
        %a = affine.load %A[%k, %i] : memref<1000x1000xf64>
        %bkj = affine.load %B[%k, %j] : memref<1000x1200xf64>
        %bij = affine.load %B[%i, %j] : memref<1000x1200xf64>
        %m = arith.mulf %a, %bkj : f64
        %s = arith.addf %bij, %m : f64
        affine.store %s, %B[%i, %j] : memref<1000x1200xf64>
      }
    }
  }
  return
}

// covariance compute nest, LARGE: M=1200 N=1400.  j runs i..M: triangular AND
// a two-reference group on `data` sharing the k index.
func.func @covariance(%data: memref<1400x1200xf64>,
                      %cov: memref<1200x1200xf64>) {
  affine.for %i = 0 to 1200 {
    affine.for %j = affine_map<(d0) -> (d0)>(%i) to 1200 {
      affine.for %k = 0 to 1400 {
        %dki = affine.load %data[%k, %i] : memref<1400x1200xf64>
        %dkj = affine.load %data[%k, %j] : memref<1400x1200xf64>
        %c = affine.load %cov[%i, %j] : memref<1200x1200xf64>
        %m = arith.mulf %dki, %dkj : f64
        %s = arith.addf %c, %m : f64
        affine.store %s, %cov[%i, %j] : memref<1200x1200xf64>
      }
    }
  }
  return
}
