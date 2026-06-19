#!/usr/bin/env python3
"""Generate a large, clean, batch-shardable affine kernel + timing/checksum
driver for the whole-kernel SPMD perf study (PARALLEL_SPMD_SPEC.md S3).

The kernel is K owner-aligned elementwise layers over a batch axis N, each
element running a P-step straight-line compute recurrence (compute-bound, so
thread scaling is not capped by memory bandwidth).  Every layer is owner-aligned
on N, so dr-par-bubbles{par-spmd} fuses them into ONE par.forall over N with
ZERO barriers -- the embarrassingly-parallel barrier-free batch SPMD case.

@main fills the input (scf, so it is not itself materialized), times R kernel
calls with rtclock, prints the per-call seconds, then prints a scalar checksum
of the output (correctness + DCE guard).

Usage: gen_spmd_perf_kernel.py [N] [M] [K] [P] [R]
"""
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 64     # batch (shard axis)
M = int(sys.argv[2]) if len(sys.argv) > 2 else 4096   # elements per batch
K = int(sys.argv[3]) if len(sys.argv) > 3 else 4      # layers
P = int(sys.argv[4]) if len(sys.argv) > 4 else 192    # compute steps / element
R = int(sys.argv[5]) if len(sys.argv) > 5 else 3      # timed repeats
# arg 6: interleave=1 emits each layer's scratch alloc BETWEEN bands (real-ONNX
# shape; exercises S2 inter-band glue hoisting).  default 0 = allocs at top.
INTERLEAVE = (len(sys.argv) > 6 and sys.argv[6] == "1")

T = f"memref<{N}x{M}xf32>"
o = []
w = o.append

w("func.func private @rtclock() -> f64")
w("func.func private @printF64(f64)")
w("func.func private @printNewline()")
w("func.func private @printMemrefF32(memref<*xf32>)")
w("")

# scratch buffers S0..S_{K-2}; layer l reads src(l), writes dst(l).
def src(l): return "%A" if l == 0 else f"%S{l-1}"
def dst(l): return "%OUT" if l == K-1 else f"%S{l}"

w(f"func.func @kernel(%A: {T}, %OUT: {T}) {{")
if not INTERLEAVE:
    for l in range(K-1):
        w(f"  %S{l} = memref.alloc() : {T}")
w("  %c1 = arith.constant 0.999 : f32")
w("  %c2 = arith.constant 0.013 : f32")
for l in range(K):
    if INTERLEAVE and l < K-1:
        w(f"  %S{l} = memref.alloc() : {T}")  # scratch declared between bands
    w(f"  affine.for %n = 0 to {N} {{")
    w(f"    affine.for %i = 0 to {M} {{")
    w(f"      %x = affine.load {src(l)}[%n, %i] : {T}")
    cur = "%x"
    for p in range(P):
        w(f"      %m{l}_{p} = arith.mulf {cur}, %c1 : f32")
        w(f"      %t{l}_{p} = arith.mulf %x, %c2 : f32")
        w(f"      %r{l}_{p} = arith.addf %m{l}_{p}, %t{l}_{p} : f32")
        cur = f"%r{l}_{p}"
    w(f"      affine.store {cur}, {dst(l)}[%n, %i] : {T}")
    w("    }")
    w("  }")
for l in range(K-1):
    w(f"  memref.dealloc %S{l} : {T}")
w("  return")
w("}")
w("")

w("func.func @main() {")
w("  %c0 = arith.constant 0 : index")
w(f"  %cN = arith.constant {N} : index")
w(f"  %cM = arith.constant {M} : index")
w(f"  %cR = arith.constant {R} : index")
w("  %c1i = arith.constant 1 : index")
w("  %c97 = arith.constant 97 : index")
w(f"  %A = memref.alloc() : {T}")
w(f"  %OUT = memref.alloc() : {T}")
# fill A[n,i] = ((n*M+i) mod 97) * 0.01  (scf -> main is not materialized)
w("  scf.for %n = %c0 to %cN step %c1i {")
w("    scf.for %i = %c0 to %cM step %c1i {")
w("      %nm = arith.muli %n, %cM : index")
w("      %idx = arith.addi %nm, %i : index")
w("      %mod = arith.remui %idx, %c97 : index")
w("      %ic = arith.index_cast %mod : index to i32")
w("      %f = arith.sitofp %ic : i32 to f32")
w("      %cscale = arith.constant 0.01 : f32")
w("      %v = arith.mulf %f, %cscale : f32")
w(f"      memref.store %v, %A[%n, %i] : {T}")
w("    }")
w("  }")
# time R kernel calls
w("  %t0 = call @rtclock() : () -> f64")
w("  scf.for %r = %c0 to %cR step %c1i {")
w(f"    func.call @kernel(%A, %OUT) : ({T}, {T}) -> ()")
w("  }")
w("  %t1 = call @rtclock() : () -> f64")
w("  %dt = arith.subf %t1, %t0 : f64")
w(f"  %rf = arith.constant {float(R)} : f64")
w("  %per = arith.divf %dt, %rf : f64")
w("  call @printF64(%per) : (f64) -> ()")
w("  call @printNewline() : () -> ()")
# checksum: sum(OUT) printed as a 1-elem memref (correctness + DCE guard)
w("  %z = arith.constant 0.0 : f32")
w("  %sum = scf.for %n = %c0 to %cN step %c1i iter_args(%a0 = %z) -> f32 {")
w("    %s = scf.for %i = %c0 to %cM step %c1i iter_args(%a1 = %a0) -> f32 {")
w(f"      %e = memref.load %OUT[%n, %i] : {T}")
w("      %na = arith.addf %a1, %e : f32")
w("      scf.yield %na : f32")
w("    }")
w("    scf.yield %s : f32")
w("  }")
w("  %ck = memref.alloc() : memref<1xf32>")
w("  memref.store %sum, %ck[%c0] : memref<1xf32>")
w("  %cku = memref.cast %ck : memref<1xf32> to memref<*xf32>")
w("  call @printMemrefF32(%cku) : (memref<*xf32>) -> ()")
w(f"  memref.dealloc %A : {T}")
w(f"  memref.dealloc %OUT : {T}")
w("  memref.dealloc %ck : memref<1xf32>")
w("  return")
w("}")

sys.stdout.write("\n".join(o) + "\n")
