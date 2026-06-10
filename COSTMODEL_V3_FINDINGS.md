# Cost Model v3 — Implementation Findings

**Date:** 2026-06-11 (overnight session). **Spec:** COSTMODEL_V3_SPEC.md.
**Checkpoints:** `costmodel_v3_1`..`costmodel_v3_6`. Numbers in the table
are host single-kernel XL runs vs `cgeist-base` (median of 3, SINK-gated).

## Campaign confirmation (results/o0-campaign-2026-06-11-v3, 29/29 kernels, zero failures)

| geomean vs cgeist-base | LARGE | EXTRALARGE |
|---|---|---|
| distribute-tile-regblock | **2.12x** | **2.05x** |
| distribute-tile-then-regblock | 1.93x | **2.05x** (and fixes gemm: 4.24x vs 1.8x) |
| polly | 1.14x | 1.42x |
| v2 best (for reference) | 1.69x | 1.61x |

Head-to-head vs Polly: **21/29 at both sizes** (v2: 18/29).  Campaign-
scale confirmations: covariance 11.39x, correlation 11.03x, trmm 17.08x,
jacobi-2d 2.86x, heat-3d 2.09x, gramschmidt 1.56x, floyd-warshall 1.34x.
Remaining XL losses are exactly the spec's deliberate skips (symm 1.01 vs
1.50, seidel-2d 1.09 vs 2.27, fdtd-2d 1.00 vs 1.35, gramschmidt 1.56 vs
2.26) plus borderline lu/durbin/nussinov/gemver.  Stop-the-line scan vs
the v2 campaign: every sub-0.97x entry is either byte-identical
pre-existing (cholesky 0.83 / lu 0.86 / trisolv 0.90 bands) or noise on
sub-10ms kernels (jacobi-1d XL runs ~1 ms; its configs are unchanged,
only the noisy baseline moved).  doitgen remains the one pre-existing
pipeline FAIL (spec non-goal).

## Headline

| kernel (XL) | v2 best | v3 best | Polly | mechanism |
|---|---|---|---|---|
| covariance | 3.70x | **11.55x** | 6.25x | A2 k-chunked peeled HEAD |
| correlation | 1.02x | **11.15x** | 1.90x | A1 remainder strip + A2 |
| trmm | 1.02x | **16.1x** | 3.73x | A3 in-place triangular peel |
| gramschmidt | 0.94x | **1.55x** | 2.25x | A5 stage 3b vectorization |
| jacobi-2d | 1.00x | **2.91x** | 1.12x | B2 skewed time-tiling |
| heat-3d | 1.01x | **2.08x** | 1.18x | B2 |
| gemm (combined cfg) | 1.81x | **4.21x** | 2.41x | A6 tile-before-regblock |

No regressions measured anywhere; every transformed config is SINK
bit-identical (except the syrk/syr2k dot-family 2-ulp reassoc tolerance,
unchanged from v2); covariance/trmm/jacobi-1d/2d/heat-3d additionally
SMALL array-dump bit-identical.

## What landed (per checkpoint)

1. **v3_1 — A1 remainder strips.** Both triangular peels now strip-mine
   only the mr-divisible prefix and clone the ORIGINAL nest as a scalar
   epilogue (original coordinates, jam-proof). Unlocked correlation
   (i-trip 2599).
2. **v3_2 — A2 k-chunking + A6 order.** The peeled HEAD's reduction is
   chunked (largest divisor of the k-trip <= `peel-k-tile`, default 128,
   floor 64) when a multiplicand is k-STRIDED and the stream exceeds the
   effective LLC. The pathology is TLB-side: data[k][j] at XL strides
   20.8 KB = a fresh page per k step; the chunk keeps the slab's page set
   resident across the j sweep. Wide plateau (targets 75–150 all
   11.0–11.5x). Dot-family (stride-1-in-k: syrk/syr2k) excluded by gate —
   measured unchanged. A6: `distribute-tile-then-regblock` (tile FIRST) is
   >= max(other orders) - 2.6% on all 8 dense kernels; regblock-first had
   capped gemm at 1.79x.
3. **v3_3 — A3 in-place triangular peel.** trmm's real shape after
   distribute (the alpha-scale legally fissions out) is i{j{k=i+1..N}}
   with B both accumulator and multiplicand. New
   `peelInPlaceTriangularInnermost`: CORNER first (intra-strip k's,
   original coords, provably skipped by Stage 3), then MAIN
   (k >= ii+mr-1+c, row-disjoint from strip accumulators — certified via
   `dr.acc_no_alias` so the alias guard steps aside). Ascending-k order
   per (i,j) preserved → bit-identical. LU stays protected (lit-pinned).
4. **v3_4 — B1 reference groups.** ReuseAnalysis now records stencil-
   neighbour groups (same memref+coeff matrix, differing const offsets)
   with per-dim span and per-loop carriesReuse. Analysis-only; pipeline
   IR bit-identical on all staged kernels. jacobi-2d/heat-3d at -O0 use
   two memrefs (NO `t % 2` subscripts) — the semi-affine mod extension is
   unnecessary for B2's targets and was not built.
5. **v3_5 — B2 `dr-affine-stencil-time-tile`.** Matches
   `for t { B<-f(A); A<-g(B) }` (halo <= 1). Half-step virtual time
   tau = 2t+phase, skew-by-tau makes all dep distances >= 0 (fully
   permutable) → rectangular tiling, emitted in original coordinates as
   max/min windows. Two load-bearing details: (a) space-tile loops cover
   the BAND-LOCAL skewed range [lo+2tt, hi+2(tt+Tt)) — the naive whole-2T
   range drowns 3-D in empty tiles (heat-3d 1.0x → 2.08x); (b) tile-s=0
   derives strip width from the cache ((Ts+2Tt)^d box <= effLLC/2),
   tile-t=0 uses measured per-rank defaults (2-D: 128, 3-D: 24).
   Footprint gate keeps jacobi-1d untouched (Polly's 0.34x mistake).
   seidel-2d does not match (in-place, single nest) — left at 1.09x-ish
   per spec B3 recommendation. fdtd-2d (4 nests, 3 arrays) deferred.
6. **v3_6 — A5 stage 3b + symbolic vl-split.** gramschmidt's 0.94x =
   distribute splits the per-k j-sweep (locality guard can't analyze the
   out-of-model band, stays optimistic) and Stage 3 can't reach the
   projection dot under the sequential k (needs a parallel sOut).
   Fix: `vectorizeBroadcastBand` learned an affine vl-split for
   symbolic-trip inner loops (mainUb = lb + ((ub-lb) floordiv VL)*VL),
   and a new Stage 3b vectorizes leftover innermost reductions whose
   parent spatial loop is parallel (no jam involved). Side effect:
   lb-triangular DIAG corners and A1 epilogues now get explicit vector
   kernels too (trmm 16.1 → 16.8x).

## Diagnoses & oddities worth knowing

- **Stage 3 `findReductionLoopUnder(func)` cross-talk:** after jamming a
  sOut, Stage 3 re-finds "the" reduction by walking the whole func — it
  can pick up a DIFFERENT (earlier-in-walk-order) nest, e.g. covariance's
  mean nest, leaving the jammed cov HEAD un-promoted (it still wins via
  LLVM scalar promotion + SLP). Not fixed tonight: every measured kernel
  is healthy, and re-plumbing Stage 3's iteration order is a refactor
  with regression surface. Tracked as a v4 cleanup candidate.
- **The harness auto-wraps pipelines** with `inline,` + `,dr-pin-liveout`
  (polybench-bench.sh:349) — host single-kernel commands must do the
  same or kernels keep symbolic bounds and nothing fires.
- **nussinov staging in /tmp/fuseinv is stale** (pre-rewrite-struct
  main_O0.mlir with `memref.alloc : !llvm.ptr`); use main_O0_fix.mlir.
- The `_lc` lean-ctx shell wrapper still mangles compound one-liners and
  multi-line commit messages (use `git commit -F file`, script files).

## Harness changes (drcc-benchmarks, uncommitted there)

- `polybench-bench.sh`: new configs `distribute-tile-then-regblock`,
  `stencil-time-tile` (llc-sharers=2).
- `run-campaign-v3.sh`: v3 config matrix, OUT_DIR
  `results/o0-campaign-2026-06-11-v3`.

## Non-goals honored

No ISL/diamond tiling, no OpenMP, no autotuning. symm (A4) skipped per
spec timebox; seidel-2d (B3) skipped per spec recommendation; fdtd-2d
match generalization deferred. doitgen pre-existing failure untouched.
