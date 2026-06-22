#!/usr/bin/env python3
# polybench_campaign_gen.py <kernel> <N> [reps]
#
# Emit a SELF-CONTAINED, CONSTANT-BOUND (size-specialized, deployment-form)
# PolyBench kernel + a timing/checksum @main on stdout. Constant bounds are the
# point: the size-parameterized (symbolic) extract blocks affine-register-block;
# the fixed-dataset deployment form does not (see commit 9e7efa8).
#
# @main: deterministic mod-prime fill (values in [0,1), no NaN/overflow), times
# `reps` calls to the kernel via @rtclock (median taken downstream), then prints
# a scalar checksum (sum over the kernel's output buffer). Correctness is
# seq==config on the checksum, so re-running the in-place kernel `reps` times is
# fine as long as every config runs the identical reps.
#
# Classes covered: contraction/BLAS-3 (gemm,2mm,syrk), BLAS-2 (mvt,atax),
# stencil (jacobi-2d,heat-3d), datamining (covariance), solver (lu).
import sys

REPS = 5


class Emit:
    """Tiny MLIR emitter with a unique-name counter and an index-constant cache."""

    def __init__(self):
        self.lines = []
        self.n = 0
        self.idx = {}   # value -> ssa for index constants
        self.consts_block = []

    def t(self, base="v"):
        self.n += 1
        return f"%{base}{self.n}"

    def emit(self, s):
        self.lines.append(s)

    def cidx(self, v):
        if v not in self.idx:
            s = f"%ci_{v}"
            self.consts_block.append(f"  {s} = arith.constant {v} : index")
            self.idx[v] = s
        return self.idx[v]

    def text(self):
        return "\n".join(self.lines)


def fill2d(e, mref, R, C):
    """Deterministic [0,1) fill of a RxC f64 memref via (i*C+j) mod 97 / 97."""
    i, j = e.t("fi"), e.t("fj")
    cR, cC = e.cidx(R), e.cidx(C)
    p97, f97 = e.cidx(97), "%cf_97"
    body = []
    tt, t2, m, mi, mf, v = (e.t("ft"), e.t("ft2"), e.t("fm"),
                            e.t("fmi"), e.t("fmf"), e.t("fv"))
    body.append(f"      {tt} = arith.muli {i}, {cC} : index")
    body.append(f"      {t2} = arith.addi {tt}, {j} : index")
    body.append(f"      {m} = arith.remui {t2}, {p97} : index")
    body.append(f"      {mi} = arith.index_cast {m} : index to i64")
    body.append(f"      {mf} = arith.sitofp {mi} : i64 to f64")
    body.append(f"      {v} = arith.divf {mf}, {f97} : f64")
    body.append(f"      memref.store {v}, {mref}[{i}, {j}] : memref<{R}x{C}xf64>")
    e.emit(f"    scf.for {i} = %ci_0 to {cR} step %ci_1 {{")
    e.emit(f"      scf.for {j} = %ci_0 to {cC} step %ci_1 {{")
    e.emit("\n".join(body))
    e.emit("    } }")


def fill1d(e, mref, R):
    i = e.t("gi")
    cR = e.cidx(R)
    p97, f97 = e.cidx(97), "%cf_97"
    m, mi, mf, v = e.t("gm"), e.t("gmi"), e.t("gmf"), e.t("gv")
    e.emit(f"    scf.for {i} = %ci_0 to {cR} step %ci_1 {{")
    e.emit(f"      {m} = arith.remui {i}, {p97} : index")
    e.emit(f"      {mi} = arith.index_cast {m} : index to i64")
    e.emit(f"      {mf} = arith.sitofp {mi} : i64 to f64")
    e.emit(f"      {v} = arith.divf {mf}, {f97} : f64")
    e.emit(f"      memref.store {v}, {mref}[{i}] : memref<{R}xf64>")
    e.emit("    }")


def checksum2d(e, mref, R, C):
    """Sum all elements of a RxC f64 memref into CK[0] and print."""
    i, j = e.t("ki"), e.t("kj")
    cR, cC = e.cidx(R), e.cidx(C)
    acc, el, na = e.t("kacc"), e.t("kel"), e.t("kna")
    e.emit(f"    scf.for {i} = %ci_0 to {cR} step %ci_1 {{")
    e.emit(f"      scf.for {j} = %ci_0 to {cC} step %ci_1 {{")
    e.emit(f"        {el} = memref.load {mref}[{i}, {j}] : memref<{R}x{C}xf64>")
    e.emit(f"        {acc} = memref.load %CK[%ci_0] : memref<1xf64>")
    e.emit(f"        {na} = arith.addf {acc}, {el} : f64")
    e.emit(f"        memref.store {na}, %CK[%ci_0] : memref<1xf64>")
    e.emit("    } }")


def checksum1d(e, mref, R):
    i = e.t("li")
    cR = e.cidx(R)
    acc, el, na = e.t("lacc"), e.t("lel"), e.t("lna")
    e.emit(f"    scf.for {i} = %ci_0 to {cR} step %ci_1 {{")
    e.emit(f"      {el} = memref.load {mref}[{i}] : memref<{R}xf64>")
    e.emit(f"      {acc} = memref.load %CK[%ci_0] : memref<1xf64>")
    e.emit(f"      {na} = arith.addf {acc}, {el} : f64")
    e.emit(f"      memref.store {na}, %CK[%ci_0] : memref<1xf64>")
    e.emit("    }")


# ---------------------------------------------------------------------------
# Kernel bodies (constant-bound). Each returns (kernel_func_text, builder) where
# builder(e) emits allocs+fills, returns (call_text, ck_emitter).
# ---------------------------------------------------------------------------

def k_gemm(N):
    kf = f"""func.func @kernel_gemm(%alpha: f64, %beta: f64, %C: memref<{N}x{N}xf64>, %A: memref<{N}x{N}xf64>, %B: memref<{N}x{N}xf64>) {{
  affine.for %i = 0 to {N} {{
    affine.for %j = 0 to {N} {{
      %c = affine.load %C[%i, %j] : memref<{N}x{N}xf64>
      %cb = arith.mulf %c, %beta : f64
      affine.store %cb, %C[%i, %j] : memref<{N}x{N}xf64>
    }}
    affine.for %k = 0 to {N} {{
      affine.for %j = 0 to {N} {{
        %a = affine.load %A[%i, %k] : memref<{N}x{N}xf64>
        %ab = arith.mulf %alpha, %a : f64
        %b = affine.load %B[%k, %j] : memref<{N}x{N}xf64>
        %abb = arith.mulf %ab, %b : f64
        %c = affine.load %C[%i, %j] : memref<{N}x{N}xf64>
        %s = arith.addf %c, %abb : f64
        affine.store %s, %C[%i, %j] : memref<{N}x{N}xf64>
      }}
    }}
  }}
  return
}}"""

    def build(e):
        e.emit(f"  %C = memref.alloc() : memref<{N}x{N}xf64>")
        e.emit(f"  %A = memref.alloc() : memref<{N}x{N}xf64>")
        e.emit(f"  %B = memref.alloc() : memref<{N}x{N}xf64>")
        fill2d(e, "%C", N, N)
        fill2d(e, "%A", N, N)
        fill2d(e, "%B", N, N)
        call = f"    func.call @kernel_gemm(%alpha, %beta, %C, %A, %B) : (f64, f64, memref<{N}x{N}xf64>, memref<{N}x{N}xf64>, memref<{N}x{N}xf64>) -> ()"
        return call, (lambda e2: checksum2d(e2, "%C", N, N)), True

    return kf, build


def k_2mm(N):
    # tmp = alpha*A*B ; D = beta*D + tmp*C
    kf = f"""func.func @kernel_2mm(%alpha: f64, %beta: f64, %tmp: memref<{N}x{N}xf64>, %A: memref<{N}x{N}xf64>, %B: memref<{N}x{N}xf64>, %C: memref<{N}x{N}xf64>, %D: memref<{N}x{N}xf64>) {{
  affine.for %i = 0 to {N} {{
    affine.for %j = 0 to {N} {{
      %z = arith.constant 0.0 : f64
      affine.store %z, %tmp[%i, %j] : memref<{N}x{N}xf64>
    }}
    affine.for %k = 0 to {N} {{
      affine.for %j = 0 to {N} {{
        %a = affine.load %A[%i, %k] : memref<{N}x{N}xf64>
        %ab = arith.mulf %alpha, %a : f64
        %b = affine.load %B[%k, %j] : memref<{N}x{N}xf64>
        %abb = arith.mulf %ab, %b : f64
        %t = affine.load %tmp[%i, %j] : memref<{N}x{N}xf64>
        %s = arith.addf %t, %abb : f64
        affine.store %s, %tmp[%i, %j] : memref<{N}x{N}xf64>
      }}
    }}
  }}
  affine.for %i = 0 to {N} {{
    affine.for %j = 0 to {N} {{
      %d = affine.load %D[%i, %j] : memref<{N}x{N}xf64>
      %db = arith.mulf %d, %beta : f64
      affine.store %db, %D[%i, %j] : memref<{N}x{N}xf64>
    }}
    affine.for %k = 0 to {N} {{
      affine.for %j = 0 to {N} {{
        %t = affine.load %tmp[%i, %k] : memref<{N}x{N}xf64>
        %c = affine.load %C[%k, %j] : memref<{N}x{N}xf64>
        %tc = arith.mulf %t, %c : f64
        %d = affine.load %D[%i, %j] : memref<{N}x{N}xf64>
        %s = arith.addf %d, %tc : f64
        affine.store %s, %D[%i, %j] : memref<{N}x{N}xf64>
      }}
    }}
  }}
  return
}}"""

    def build(e):
        for nm in ("%tmp", "%A", "%B", "%C", "%D"):
            e.emit(f"  {nm} = memref.alloc() : memref<{N}x{N}xf64>")
        fill2d(e, "%A", N, N)
        fill2d(e, "%B", N, N)
        fill2d(e, "%C", N, N)
        fill2d(e, "%D", N, N)
        call = f"    func.call @kernel_2mm(%alpha, %beta, %tmp, %A, %B, %C, %D) : (f64, f64, memref<{N}x{N}xf64>, memref<{N}x{N}xf64>, memref<{N}x{N}xf64>, memref<{N}x{N}xf64>, memref<{N}x{N}xf64>) -> ()"
        return call, (lambda e2: checksum2d(e2, "%D", N, N)), True

    return kf, build


def k_syrk(N):
    # C = beta*C + alpha*A*A^T, lower triangular (j <= i). k-INNERMOST so the
    # baseline is cache-fair (A[i,k]/A[j,k] are contiguous row sweeps, C[i,j] is
    # a scalar reduction) -- the polybench-reference k-outer order reloads C[i,j]
    # N times and would inflate the codegen speedup against a strawman baseline.
    kf = f"""func.func @kernel_syrk(%alpha: f64, %beta: f64, %C: memref<{N}x{N}xf64>, %A: memref<{N}x{N}xf64>) {{
  affine.for %i = 0 to {N} {{
    affine.for %j = 0 to #map_ip1(%i) {{
      %c = affine.load %C[%i, %j] : memref<{N}x{N}xf64>
      %cb = arith.mulf %c, %beta : f64
      affine.store %cb, %C[%i, %j] : memref<{N}x{N}xf64>
    }}
    affine.for %j = 0 to #map_ip1(%i) {{
      affine.for %k = 0 to {N} {{
        %a = affine.load %A[%i, %k] : memref<{N}x{N}xf64>
        %aa = affine.load %A[%j, %k] : memref<{N}x{N}xf64>
        %p = arith.mulf %a, %aa : f64
        %ap = arith.mulf %alpha, %p : f64
        %c = affine.load %C[%i, %j] : memref<{N}x{N}xf64>
        %s = arith.addf %c, %ap : f64
        affine.store %s, %C[%i, %j] : memref<{N}x{N}xf64>
      }}
    }}
  }}
  return
}}"""

    def build(e):
        e.emit(f"  %C = memref.alloc() : memref<{N}x{N}xf64>")
        e.emit(f"  %A = memref.alloc() : memref<{N}x{N}xf64>")
        fill2d(e, "%C", N, N)
        fill2d(e, "%A", N, N)
        call = f"    func.call @kernel_syrk(%alpha, %beta, %C, %A) : (f64, f64, memref<{N}x{N}xf64>, memref<{N}x{N}xf64>) -> ()"
        return call, (lambda e2: checksum2d(e2, "%C", N, N)), True

    # affine map for triangular bound j in [0, i+1)
    return ("#map_ip1 = affine_map<(d0) -> (d0 + 1)>\n" + kf), build


def k_mvt(N):
    # x1 += A*y1 ; x2 += A^T*y2   (two matvecs, row-parallel)
    kf = f"""func.func @kernel_mvt(%x1: memref<{N}xf64>, %x2: memref<{N}xf64>, %y1: memref<{N}xf64>, %y2: memref<{N}xf64>, %A: memref<{N}x{N}xf64>) {{
  affine.for %i = 0 to {N} {{
    affine.for %j = 0 to {N} {{
      %a = affine.load %A[%i, %j] : memref<{N}x{N}xf64>
      %y = affine.load %y1[%j] : memref<{N}xf64>
      %ay = arith.mulf %a, %y : f64
      %x = affine.load %x1[%i] : memref<{N}xf64>
      %s = arith.addf %x, %ay : f64
      affine.store %s, %x1[%i] : memref<{N}xf64>
    }}
  }}
  affine.for %i = 0 to {N} {{
    affine.for %j = 0 to {N} {{
      %a = affine.load %A[%j, %i] : memref<{N}x{N}xf64>
      %y = affine.load %y2[%j] : memref<{N}xf64>
      %ay = arith.mulf %a, %y : f64
      %x = affine.load %x2[%i] : memref<{N}xf64>
      %s = arith.addf %x, %ay : f64
      affine.store %s, %x2[%i] : memref<{N}xf64>
    }}
  }}
  return
}}"""

    def build(e):
        for nm in ("%x1", "%x2", "%y1", "%y2"):
            e.emit(f"  {nm} = memref.alloc() : memref<{N}xf64>")
        e.emit(f"  %A = memref.alloc() : memref<{N}x{N}xf64>")
        fill1d(e, "%x1", N)
        fill1d(e, "%x2", N)
        fill1d(e, "%y1", N)
        fill1d(e, "%y2", N)
        fill2d(e, "%A", N, N)
        call = f"    func.call @kernel_mvt(%x1, %x2, %y1, %y2, %A) : (memref<{N}xf64>, memref<{N}xf64>, memref<{N}xf64>, memref<{N}xf64>, memref<{N}x{N}xf64>) -> ()"
        return call, (lambda e2: checksum1d(e2, "%x1", N)), True

    return kf, build


def k_atax(N):
    # tmp = A*x ; y = A^T*tmp
    kf = f"""func.func @kernel_atax(%A: memref<{N}x{N}xf64>, %x: memref<{N}xf64>, %y: memref<{N}xf64>, %tmp: memref<{N}xf64>) {{
  affine.for %i = 0 to {N} {{
    %z = arith.constant 0.0 : f64
    affine.store %z, %tmp[%i] : memref<{N}xf64>
    affine.for %j = 0 to {N} {{
      %a = affine.load %A[%i, %j] : memref<{N}x{N}xf64>
      %xx = affine.load %x[%j] : memref<{N}xf64>
      %ax = arith.mulf %a, %xx : f64
      %t = affine.load %tmp[%i] : memref<{N}xf64>
      %s = arith.addf %t, %ax : f64
      affine.store %s, %tmp[%i] : memref<{N}xf64>
    }}
  }}
  affine.for %i = 0 to {N} {{
    %z = arith.constant 0.0 : f64
    affine.store %z, %y[%i] : memref<{N}xf64>
  }}
  affine.for %i = 0 to {N} {{
    affine.for %j = 0 to {N} {{
      %a = affine.load %A[%i, %j] : memref<{N}x{N}xf64>
      %t = affine.load %tmp[%i] : memref<{N}xf64>
      %at = arith.mulf %a, %t : f64
      %y0 = affine.load %y[%j] : memref<{N}xf64>
      %s = arith.addf %y0, %at : f64
      affine.store %s, %y[%j] : memref<{N}xf64>
    }}
  }}
  return
}}"""

    def build(e):
        e.emit(f"  %A = memref.alloc() : memref<{N}x{N}xf64>")
        for nm in ("%x", "%y", "%tmp"):
            e.emit(f"  {nm} = memref.alloc() : memref<{N}xf64>")
        fill2d(e, "%A", N, N)
        fill1d(e, "%x", N)
        call = f"    func.call @kernel_atax(%A, %x, %y, %tmp) : (memref<{N}x{N}xf64>, memref<{N}xf64>, memref<{N}xf64>, memref<{N}xf64>) -> ()"
        return call, (lambda e2: checksum1d(e2, "%y", N)), True

    return kf, build


def k_jacobi_2d(N, T):
    # T steps ping-pong; interior rows/cols parallel -> SEQWRAP
    kf = f"""func.func @kernel_jacobi_2d(%A: memref<{N}x{N}xf64>, %B: memref<{N}x{N}xf64>) {{
  affine.for %t = 0 to {T} {{
    affine.for %i = 1 to {N-1} {{
      affine.for %j = 1 to {N-1} {{
        %c = affine.load %A[%i, %j] : memref<{N}x{N}xf64>
        %w = affine.load %A[%i, %j - 1] : memref<{N}x{N}xf64>
        %e = affine.load %A[%i, %j + 1] : memref<{N}x{N}xf64>
        %n = affine.load %A[%i - 1, %j] : memref<{N}x{N}xf64>
        %s = affine.load %A[%i + 1, %j] : memref<{N}x{N}xf64>
        %s1 = arith.addf %c, %w : f64
        %s2 = arith.addf %s1, %e : f64
        %s3 = arith.addf %s2, %n : f64
        %s4 = arith.addf %s3, %s : f64
        %f = arith.constant 0.2 : f64
        %r = arith.mulf %s4, %f : f64
        affine.store %r, %B[%i, %j] : memref<{N}x{N}xf64>
      }}
    }}
    affine.for %i = 1 to {N-1} {{
      affine.for %j = 1 to {N-1} {{
        %c = affine.load %B[%i, %j] : memref<{N}x{N}xf64>
        %w = affine.load %B[%i, %j - 1] : memref<{N}x{N}xf64>
        %e = affine.load %B[%i, %j + 1] : memref<{N}x{N}xf64>
        %n = affine.load %B[%i - 1, %j] : memref<{N}x{N}xf64>
        %s = affine.load %B[%i + 1, %j] : memref<{N}x{N}xf64>
        %s1 = arith.addf %c, %w : f64
        %s2 = arith.addf %s1, %e : f64
        %s3 = arith.addf %s2, %n : f64
        %s4 = arith.addf %s3, %s : f64
        %f = arith.constant 0.2 : f64
        %r = arith.mulf %s4, %f : f64
        affine.store %r, %A[%i, %j] : memref<{N}x{N}xf64>
      }}
    }}
  }}
  return
}}"""

    def build(e):
        e.emit(f"  %A = memref.alloc() : memref<{N}x{N}xf64>")
        e.emit(f"  %B = memref.alloc() : memref<{N}x{N}xf64>")
        fill2d(e, "%A", N, N)
        fill2d(e, "%B", N, N)
        call = f"    func.call @kernel_jacobi_2d(%A, %B) : (memref<{N}x{N}xf64>, memref<{N}x{N}xf64>) -> ()"
        return call, (lambda e2: checksum2d(e2, "%A", N, N)), True

    return kf, build


def k_heat_3d(N, T):
    kf = f"""func.func @kernel_heat_3d(%A: memref<{N}x{N}x{N}xf64>, %B: memref<{N}x{N}x{N}xf64>) {{
  affine.for %t = 0 to {T} {{
    affine.for %i = 1 to {N-1} {{
      affine.for %j = 1 to {N-1} {{
        affine.for %k = 1 to {N-1} {{
          %c = affine.load %A[%i, %j, %k] : memref<{N}x{N}x{N}xf64>
          %xp = affine.load %A[%i + 1, %j, %k] : memref<{N}x{N}x{N}xf64>
          %xm = affine.load %A[%i - 1, %j, %k] : memref<{N}x{N}x{N}xf64>
          %yp = affine.load %A[%i, %j + 1, %k] : memref<{N}x{N}x{N}xf64>
          %ym = affine.load %A[%i, %j - 1, %k] : memref<{N}x{N}x{N}xf64>
          %zp = affine.load %A[%i, %j, %k + 1] : memref<{N}x{N}x{N}xf64>
          %zm = affine.load %A[%i, %j, %k - 1] : memref<{N}x{N}x{N}xf64>
          %two = arith.constant 2.0 : f64
          %ax = arith.addf %xp, %xm : f64
          %c2x = arith.mulf %two, %c : f64
          %sx = arith.subf %ax, %c2x : f64
          %ay = arith.addf %yp, %ym : f64
          %sy = arith.subf %ay, %c2x : f64
          %az = arith.addf %zp, %zm : f64
          %sz = arith.subf %az, %c2x : f64
          %h = arith.constant 0.125 : f64
          %t1 = arith.addf %sx, %sy : f64
          %t2 = arith.addf %t1, %sz : f64
          %t3 = arith.mulf %h, %t2 : f64
          %r = arith.addf %c, %t3 : f64
          affine.store %r, %B[%i, %j, %k] : memref<{N}x{N}x{N}xf64>
        }}
      }}
    }}
    affine.for %i = 1 to {N-1} {{
      affine.for %j = 1 to {N-1} {{
        affine.for %k = 1 to {N-1} {{
          %c = affine.load %B[%i, %j, %k] : memref<{N}x{N}x{N}xf64>
          %xp = affine.load %B[%i + 1, %j, %k] : memref<{N}x{N}x{N}xf64>
          %xm = affine.load %B[%i - 1, %j, %k] : memref<{N}x{N}x{N}xf64>
          %yp = affine.load %B[%i, %j + 1, %k] : memref<{N}x{N}x{N}xf64>
          %ym = affine.load %B[%i, %j - 1, %k] : memref<{N}x{N}x{N}xf64>
          %zp = affine.load %B[%i, %j, %k + 1] : memref<{N}x{N}x{N}xf64>
          %zm = affine.load %B[%i, %j, %k - 1] : memref<{N}x{N}x{N}xf64>
          %two = arith.constant 2.0 : f64
          %ax = arith.addf %xp, %xm : f64
          %c2x = arith.mulf %two, %c : f64
          %sx = arith.subf %ax, %c2x : f64
          %ay = arith.addf %yp, %ym : f64
          %sy = arith.subf %ay, %c2x : f64
          %az = arith.addf %zp, %zm : f64
          %sz = arith.subf %az, %c2x : f64
          %h = arith.constant 0.125 : f64
          %t1 = arith.addf %sx, %sy : f64
          %t2 = arith.addf %t1, %sz : f64
          %t3 = arith.mulf %h, %t2 : f64
          %r = arith.addf %c, %t3 : f64
          affine.store %r, %A[%i, %j, %k] : memref<{N}x{N}x{N}xf64>
        }}
      }}
    }}
  }}
  return
}}"""

    def build(e):
        e.emit(f"  %A = memref.alloc() : memref<{N}x{N}x{N}xf64>")
        e.emit(f"  %B = memref.alloc() : memref<{N}x{N}x{N}xf64>")
        # fill 3D via collapse: simple nested fill
        for mref in ("%A", "%B"):
            i, j, k = e.t("hi"), e.t("hj"), e.t("hk")
            cN = e.cidx(N)
            p97 = e.cidx(97)
            tt, t2, t3, m, mi, mf, v = (e.t("ht"), e.t("ht2"), e.t("ht3"),
                                        e.t("hm"), e.t("hmi"), e.t("hmf"), e.t("hv"))
            e.emit(f"  scf.for {i} = %ci_0 to {cN} step %ci_1 {{")
            e.emit(f"   scf.for {j} = %ci_0 to {cN} step %ci_1 {{")
            e.emit(f"    scf.for {k} = %ci_0 to {cN} step %ci_1 {{")
            e.emit(f"      {tt} = arith.muli {i}, {cN} : index")
            e.emit(f"      {t2} = arith.addi {tt}, {j} : index")
            e.emit(f"      {t3} = arith.addi {t2}, {k} : index")
            e.emit(f"      {m} = arith.remui {t3}, {p97} : index")
            e.emit(f"      {mi} = arith.index_cast {m} : index to i64")
            e.emit(f"      {mf} = arith.sitofp {mi} : i64 to f64")
            e.emit(f"      {v} = arith.divf {mf}, %cf_97 : f64")
            e.emit(f"      memref.store {v}, {mref}[{i}, {j}, {k}] : memref<{N}x{N}x{N}xf64>")
            e.emit("   } } }")

        def ck(e2):
            i, j, k = e2.t("ci"), e2.t("cj"), e2.t("ck")
            cN = e2.cidx(N)
            el, acc, na = e2.t("cel"), e2.t("cac"), e2.t("cna")
            e2.emit(f"    scf.for {i} = %ci_0 to {cN} step %ci_1 {{")
            e2.emit(f"     scf.for {j} = %ci_0 to {cN} step %ci_1 {{")
            e2.emit(f"      scf.for {k} = %ci_0 to {cN} step %ci_1 {{")
            e2.emit(f"        {el} = memref.load %A[{i}, {j}, {k}] : memref<{N}x{N}x{N}xf64>")
            e2.emit(f"        {acc} = memref.load %CK[%ci_0] : memref<1xf64>")
            e2.emit(f"        {na} = arith.addf {acc}, {el} : f64")
            e2.emit(f"        memref.store {na}, %CK[%ci_0] : memref<1xf64>")
            e2.emit("     } } }")

        call = f"    func.call @kernel_heat_3d(%A, %B) : (memref<{N}x{N}x{N}xf64>, memref<{N}x{N}x{N}xf64>) -> ()"
        return call, ck, True

    return kf, build


def k_covariance(N, M):
    # mean -> center -> cov (symmetric). data: NxM (N samples, M features)
    kf = f"""func.func @kernel_covariance(%float_n: f64, %data: memref<{N}x{M}xf64>, %cov: memref<{M}x{M}xf64>, %mean: memref<{M}xf64>) {{
  affine.for %j = 0 to {M} {{
    %z = arith.constant 0.0 : f64
    affine.store %z, %mean[%j] : memref<{M}xf64>
    affine.for %i = 0 to {N} {{
      %d = affine.load %data[%i, %j] : memref<{N}x{M}xf64>
      %m = affine.load %mean[%j] : memref<{M}xf64>
      %s = arith.addf %m, %d : f64
      affine.store %s, %mean[%j] : memref<{M}xf64>
    }}
    %m = affine.load %mean[%j] : memref<{M}xf64>
    %md = arith.divf %m, %float_n : f64
    affine.store %md, %mean[%j] : memref<{M}xf64>
  }}
  affine.for %i = 0 to {N} {{
    affine.for %j = 0 to {M} {{
      %d = affine.load %data[%i, %j] : memref<{N}x{M}xf64>
      %m = affine.load %mean[%j] : memref<{M}xf64>
      %c = arith.subf %d, %m : f64
      affine.store %c, %data[%i, %j] : memref<{N}x{M}xf64>
    }}
  }}
  affine.for %i = 0 to {M} {{
    affine.for %j = 0 to {M} {{
      %z = arith.constant 0.0 : f64
      affine.store %z, %cov[%i, %j] : memref<{M}x{M}xf64>
    }}
    affine.for %k = 0 to {N} {{
      %a = affine.load %data[%k, %i] : memref<{N}x{M}xf64>
      affine.for %j = 0 to {M} {{
        %b = affine.load %data[%k, %j] : memref<{N}x{M}xf64>
        %p = arith.mulf %a, %b : f64
        %c = affine.load %cov[%i, %j] : memref<{M}x{M}xf64>
        %s = arith.addf %c, %p : f64
        affine.store %s, %cov[%i, %j] : memref<{M}x{M}xf64>
      }}
    }}
  }}
  return
}}"""

    def build(e):
        e.emit(f"  %data = memref.alloc() : memref<{N}x{M}xf64>")
        e.emit(f"  %cov = memref.alloc() : memref<{M}x{M}xf64>")
        e.emit(f"  %mean = memref.alloc() : memref<{M}xf64>")
        fill2d(e, "%data", N, M)
        e.emit(f"  %float_n = arith.constant {float(N)} : f64")
        call = f"    func.call @kernel_covariance(%float_n, %data, %cov, %mean) : (f64, memref<{N}x{M}xf64>, memref<{M}x{M}xf64>, memref<{M}xf64>) -> ()"
        return call, (lambda e2: checksum2d(e2, "%cov", M, M)), True

    return kf, build


def k_lu(N):
    # LU decomposition (no pivot) -- inherently sequential dependences
    kf = f"""func.func @kernel_lu(%A: memref<{N}x{N}xf64>) {{
  affine.for %i = 0 to {N} {{
    affine.for %j = 0 to #map_id(%i) {{
      affine.for %k = 0 to #map_id(%j) {{
        %aik = affine.load %A[%i, %k] : memref<{N}x{N}xf64>
        %akj = affine.load %A[%k, %j] : memref<{N}x{N}xf64>
        %p = arith.mulf %aik, %akj : f64
        %aij = affine.load %A[%i, %j] : memref<{N}x{N}xf64>
        %s = arith.subf %aij, %p : f64
        affine.store %s, %A[%i, %j] : memref<{N}x{N}xf64>
      }}
      %ajj = affine.load %A[%j, %j] : memref<{N}x{N}xf64>
      %aij = affine.load %A[%i, %j] : memref<{N}x{N}xf64>
      %d = arith.divf %aij, %ajj : f64
      affine.store %d, %A[%i, %j] : memref<{N}x{N}xf64>
    }}
    affine.for %j = #map_id(%i) to {N} {{
      affine.for %k = 0 to #map_id(%i) {{
        %aik = affine.load %A[%i, %k] : memref<{N}x{N}xf64>
        %akj = affine.load %A[%k, %j] : memref<{N}x{N}xf64>
        %p = arith.mulf %aik, %akj : f64
        %aij = affine.load %A[%i, %j] : memref<{N}x{N}xf64>
        %s = arith.subf %aij, %p : f64
        affine.store %s, %A[%i, %j] : memref<{N}x{N}xf64>
      }}
    }}
  }}
  return
}}"""

    def build(e):
        e.emit(f"  %A = memref.alloc() : memref<{N}x{N}xf64>")
        # diagonally-dominant fill so divisions are stable
        i, j = e.t("di"), e.t("dj")
        cN = e.cidx(N)
        p97 = e.cidx(97)
        tt, t2, m, mi, mf, base, isdiag, dv, fv = (
            e.t("dt"), e.t("dt2"), e.t("dm"), e.t("dmi"), e.t("dmf"),
            e.t("dbase"), e.t("ddi"), e.t("ddv"), e.t("dfv"))
        e.emit(f"  scf.for {i} = %ci_0 to {cN} step %ci_1 {{")
        e.emit(f"   scf.for {j} = %ci_0 to {cN} step %ci_1 {{")
        e.emit(f"      {tt} = arith.muli {i}, {cN} : index")
        e.emit(f"      {t2} = arith.addi {tt}, {j} : index")
        e.emit(f"      {m} = arith.remui {t2}, {p97} : index")
        e.emit(f"      {mi} = arith.index_cast {m} : index to i64")
        e.emit(f"      {mf} = arith.sitofp {mi} : i64 to f64")
        e.emit(f"      {base} = arith.divf {mf}, %cf_97 : f64")
        # add N to diagonal: if i==j store base + N else base
        e.emit(f"      {isdiag} = arith.cmpi eq, {i}, {j} : index")
        e.emit(f"      %cf_N_lu = arith.constant {float(N)} : f64")
        e.emit(f"      {dv} = arith.addf {base}, %cf_N_lu : f64")
        e.emit(f"      {fv} = arith.select {isdiag}, {dv}, {base} : f64")
        e.emit(f"      memref.store {fv}, %A[{i}, {j}] : memref<{N}x{N}xf64>")
        e.emit("   } }")
        call = f"    func.call @kernel_lu(%A) : (memref<{N}x{N}xf64>) -> ()"
        return call, (lambda e2: checksum2d(e2, "%A", N, N)), True

    return ("#map_id = affine_map<(d0) -> (d0)>\n" + kf), build


KERNELS = {
    "gemm": lambda N: k_gemm(N),
    "2mm": lambda N: k_2mm(N),
    "syrk": lambda N: k_syrk(N),
    "mvt": lambda N: k_mvt(N),
    "atax": lambda N: k_atax(N),
    "jacobi-2d": lambda N: k_jacobi_2d(N, 50),
    "heat-3d": lambda N: k_heat_3d(N, 40),
    "covariance": lambda N: k_covariance(N, N),
    "lu": lambda N: k_lu(N),
}


def main():
    kernel = sys.argv[1]
    N = int(sys.argv[2])
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else REPS
    kf, build = KERNELS[kernel](N)

    e = Emit()
    # alpha/beta scalars available for kernels that use them
    e.emit("  %alpha = arith.constant 1.5 : f64")
    e.emit("  %beta = arith.constant 1.2 : f64")
    call, ck_emit, _ = build(e)

    # warmup call (untimed) -- first-touch page faults + JIT warmup off the
    # timing path so avg/stddev are not polluted by a cold outlier. seq and the
    # parallel/blocked configs all run the identical warmup+reps, so the
    # checksum (taken after all calls) stays config-invariant.
    e.emit(call)
    # timing loop
    creps = e.cidx(reps)
    e.emit(f"  %T = memref.alloc() : memref<{reps}xf64>")
    e.emit(f"  scf.for %r = %ci_0 to {creps} step %ci_1 {{")
    e.emit("    %t0 = func.call @rtclock() : () -> f64")
    e.emit(call)
    e.emit("    %t1 = func.call @rtclock() : () -> f64")
    e.emit("    %dt = arith.subf %t1, %t0 : f64")
    e.emit(f"    memref.store %dt, %T[%r] : memref<{reps}xf64>")
    e.emit("  }")
    e.emit(f"  %Tu = memref.cast %T : memref<{reps}xf64> to memref<*xf64>")
    e.emit("  func.call @printMemrefF64(%Tu) : (memref<*xf64>) -> ()")
    # checksum
    e.emit("  %CK = memref.alloc() : memref<1xf64>")
    e.emit("  %zck = arith.constant 0.0 : f64")
    e.emit("  memref.store %zck, %CK[%ci_0] : memref<1xf64>")
    ck_emit(e)
    e.emit("  %CKu = memref.cast %CK : memref<1xf64> to memref<*xf64>")
    e.emit("  func.call @printMemrefF64(%CKu) : (memref<*xf64>) -> ()")
    e.emit("  return")

    # index-constant prelude must include the always-used 0/1
    e.cidx(0)
    e.cidx(1)
    prelude = "\n".join(e.consts_block)

    print("func.func private @printMemrefF64(memref<*xf64>)")
    print("func.func private @rtclock() -> f64")
    print(kf)
    print("func.func @main() {")
    print("  %cf_97 = arith.constant 97.0 : f64")
    print(prelude)
    print(e.text())
    print("}")


if __name__ == "__main__":
    main()
