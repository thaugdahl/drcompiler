# PolyBench parallelization sweep — opportunity map + measured speedup

**Date:** 2026-06-22
**Author:** Claude (Opus 4.8)
**Ask (user):** "Focus on PolyBench — make them fast, and work on finding
parallelization opportunities across all classes in PolyBench."

**Result.** Ran the `dr-par-bubbles` parallel-axis oracle over **all 30
PolyBench/C 4.2.1 kernels** (6 classes) to map where parallelism exists, then
**measured real OpenMP thread-scaling** (1→16 threads, checksum-verified
seq≡omp) on **11 representatives spanning every class**. Headlines: dense
contractions and stencils scale near-linearly to super-linearly (**gemm 14.4×,
2mm 14.8×, jacobi-2d 18.8×, heat-3d 17.2×** at 16 threads); triangular/gram
kernels ~7×; solvers and memory-bound BLAS-2 are limited or regress (honest
negatives below). **23/30 kernels expose ≥1 parallel axis; 22/30 have a whole-
function shard axis.** The fully-sequential 7 (symm, cholesky, ludcmp, trisolv,
seidel-2d, floyd-warshall, nussinov) are correctly declined — genuine carried
dependences / recurrences / DP, not analysis conservatism.

All times are `mlir-runner --O3` median on a 16-core Zen4 (Ryzen 7950X3D),
`OMP_PROC_BIND=close OMP_PLACES=cores`. Speedup = seq(1 thread, no OpenMP) /
omp(N threads). `correct = MATCH` means the parallel checksum equals the
sequential checksum to <1e-9 relative.

---

## Pipeline

```
 PolyBench .c  --docker cgeist (LLVM18)-->  per-kernel affine MLIR  (bench/polybench-mlir/)
   analysis :  dr-opt dr-par-bubbles{par-test-diagnostics}     -> PARALLEL/SEQUENTIAL per loop
   lowering :  mlir-opt --affine-parallelize=max-nested=1       -> parallelize outermost proven axis
               --lower-affine --convert-scf-to-openmp --canonicalize ... --convert-openmp-to-llvm
   run      :  mlir-runner --O3  (libomp)                       -> thread sweep + checksum
```

The **analysis** is the novel piece: the `ParAliasOracle` (Tier 0 allocation-root
provenance, Tier 1 polyhedral `checkMemrefAccessDependence`, Tier 2 conservative)
classifies every loop axis. The **lowering** uses the stock upstream
`affine-parallelize` (whose independent dependence check agrees with the oracle
on every representative) so the measured speedup is attributable to the
parallelism itself, not to a bespoke code path. `--canonicalize` after
`convert-scf-to-openmp` is required to drop the `memref.alloca_scope` that
otherwise blocks `convert-scf-to-cf`.

---

## Opportunity map — all 30 kernels (`dr-par-bubbles{par-test-diagnostics}`)

`PAR`/`SEQ` = count of parallel / sequential loop axes. `shard` = whole-function
SPMD shard axis (`par-test-spmd`): `bands=X/Y` means X of Y top-level bands are
materializable on the chosen axis.

| class | kernel | PAR | SEQ | shard | class | kernel | PAR | SEQ | shard |
|---|---|---|---|---|---|---|---|---|---|
| datamining | correlation | 6 | 3 | 3/5 | kernels | doitgen | 2 | 3 | 2/2 |
| datamining | covariance | 5 | 2 | 3/3 | kernels | mvt | 2 | 2 | 2/2 |
| blas | gemm | 3 | 1 | 2/2 | solvers | cholesky | 0 | 4 | — |
| blas | gemver | 5 | 2 | 4/4 | solvers | durbin | 2 | 2 | 2/2 |
| blas | gesummv | 1 | 1 | 1/1 | solvers | gramschmidt | 3 | 3 | 2/3 |
| blas | symm | 0 | 3 | — | solvers | ludcmp | 0 | 9 | — |
| blas | syr2k | 3 | 1 | 1/2 | solvers | lu | 1 | 4 | 1/1 |
| blas | syrk | 3 | 1 | 1/2 | solvers | trisolv | 0 | 2 | — |
| blas | trmm | 1 | 2 | — | medley | deriche | 4 | 8 | 2/2 |
| kernels | 2mm | 4 | 2 | 2/2 | medley | floyd-warshall | 0 | 3 | — |
| kernels | 3mm | 6 | 3 | 3/3 | medley | nussinov | 0 | 3 | — (non-affine) |
| kernels | atax | 2 | 2 | 2/2 | stencils | adi | 2 | 5 | 2/2 |
| kernels | bicg | 1 | 2 | 1/1 | stencils | fdtd-2d | 7 | 1 | 2/4 |
| stencils | heat-3d | 6 | 1 | 2/2 | stencils | jacobi-1d | 2 | 1 | 2/2 |
| stencils | jacobi-2d | 4 | 1 | 2/2 | stencils | seidel-2d | 0 | 3 | — |

## Measured OpenMP scaling — 11 representatives (all 6 classes)

| kernel | class | size | seq O3 (s) | 2t | 4t | 8t | **16t** | correct |
|---|---|---|---|---|---|---|---|---|
| gemm       | blas (contraction)   | 1000³ish | 0.142 | 2.04× | 3.85× | 7.00× | **14.41×** | MATCH |
| 2mm        | kernels (2× GEMM)    | 800–1200 | 0.953 | 1.89× | 3.75× | 7.37× | **14.76×** | MATCH |
| syrk       | blas (rank-k, tri)   | 1200/1000| 0.261 | 1.15× | 1.91× | 3.59× | **7.15×**  | MATCH |
| covariance | datamining (gram)    | 1400×1200| 0.834 | 1.26× | 2.13× | 3.99× | **7.56×**  | MATCH |
| doitgen    | kernels (tensor)     | 220/140/160 | 0.359 | 1.54× | 2.64× | 3.58× | 2.36× | MATCH |
| jacobi-2d  | stencils (5-pt)      | 1300², 40t | 0.055 | 2.80× | 5.55× | 9.60× | **18.79×** | MATCH |
| heat-3d    | stencils (7-pt)      | 120³, 40t  | 0.992 | 2.85× | 5.44× | 10.44×| **17.19×** | MATCH |
| lu         | solvers (factor)     | 1000²    | 0.199 | 1.35× | 1.60× | 1.75× | 1.82×  | MATCH |
| deriche    | medley (IIR filter)  | 4096×2160| 0.079 | 1.02× | 1.00× | 0.99× | 0.87×  | MATCH |
| mvt        | kernels (2× matvec)  | 2000²    | 0.0027| 1.10× | 2.19× | 4.26× | 8.22×  | MATCH |
| atax       | kernels (A·x, Aᵀ·y)  | 1800×2100| 0.0030| 0.88× | 0.86× | 0.79× | 0.52×  | MATCH |

## Per-class reading (honest)

- **Dense contraction (gemm, 2mm, 3mm) — near-linear.** Outer spatial loop is a
  clean parallel axis; the k-reduction stays sequential within each thread. 14–15×
  at 16 threads (≈92% efficiency). The strongest, cleanest class.
- **Triangular / gram (syrk, syr2k, covariance, correlation) — ~7×.** The outer
  parallel loop has an IV-dependent trip count (`j:0..i`), so a static OpenMP
  schedule gives load imbalance; ~7× is the imbalance ceiling, not a correctness
  or analysis limit (dynamic scheduling / triangular blocking would recover it).
- **Stencils (jacobi-2d, heat-3d, jacobi-1d, fdtd-2d) — super-linear.** The time
  loop is sequential; each timestep's spatial sweep parallelizes. 17–19× at 16
  threads — super-linear because per-thread working sets fit cache. Best class.
- **Solvers (lu) — limited (1.8×).** Only the inner row-update parallelizes; the
  k-sweep carries a dependence. ludcmp/cholesky/trisolv have **no** parallel axis
  at all (declined, correctly). Blocked-factorization is a different technique.
- **Memory-bound BLAS-2 (mvt 8×, atax 0.5×).** Parallel but bandwidth-bound and
  tiny (matvec runs in ~3 ms). mvt still rides bandwidth to 8×; **atax regresses
  to 0.52×** — the work is too small for OpenMP fork/join + bandwidth saturation.
  A real negative, reported as such.
- **Medley (deriche ~1.0×).** Recursive IIR filters serialize along the filter
  direction; only the orthogonal axis parallelizes, and the kernel is memory-
  bound — no win. floyd-warshall and nussinov are sequential/non-affine DP.
- **doitgen — peaks 3.6× then regresses.** The natural parallel axes (nr, nq) are
  blocked by a **shared `sum[np]` scratch buffer** (a function argument reused
  every (r,q)); only a small inner loop parallelizes. Privatizing `sum` per
  thread is the fix — a clear, scoped opportunity.

## Key technical finding + fix — bringing PolyBench under the in-house SPMD path

The proven whole-kernel SPMD materializer (`dr-par-bubbles{par-spmd-perband}` →
`convert-par-to-omp`, validated on ResNet-50 / MNIST) originally **declined every
PolyBench kernel** — it emitted `par.critical` (serial), not `par.forall`.

**Real root cause** (an earlier draft mis-attributed this to Tier-0 allocation-
root provenance — wrong): `bandMaterializable` required a **perfect loop nest
with a straight-line innermost body** (only affine.load/store + pure ops). The
oracle correctly classifies the outer spatial loop as PARALLEL, but a PolyBench
contraction's parallel `i`-loop holds an **imperfect body** — sibling sub-nests
(e.g. gemm's beta-scale row, then the k/j accumulation) — so the perfect-nest
gate rejected it and the band fell back to `par.critical`. Fissioning the init
loops did not help because the blocker was nest *shape*, not provenance.

**Fix (landed).** When the shard loop is oracle-PARALLEL but its body is
imperfect, **de-affine the whole body into the `par.forall`** (`affine.for` →
`scf.for` with expanded bounds, `affine.load/store` → `memref`), gated by a
`deAffinable` pre-check (no iter-arg reductions / `affine.if` / unknown region
ops). Soundness is owner-computes: the oracle's PARALLEL verdict means distinct
shard iterations touch disjoint memory, so the body runs sequentially per shard
unchanged. (`lib/Transforms/ParBubbles.cpp`; lit `spmd-perband-imperfect.mlir`.)

**Result — contraction / gram / BLAS-2 now materialize under the in-house par
dialect** (`dr-par-bubbles{par-spmd-perband}` → `convert-par-to-omp`), numerically
MATCH, same speedups as the stock stand-in: gemm 13.4×, 2mm 14.2×, syrk 7.0×,
covariance 7.6×, mvt 12.8× @16t. ResNet-50 / MNIST re-validated **norm_rel_err =
0.000e+00** (resnet50 materialization counts unchanged: 82 forall / 55 critical),
full lit 241/0.

**Stencils now land too (SEQWRAP, also landed).** A band whose outermost loop is
*sequential* (the time step) wrapping parallel inner bands is materialized as one
team holding `scf.for(t) { par.forall ; par.forall }` — the team persists across
timesteps and re-distributes the spatial work each step. No `par.barrier` (it may
not nest under `scf.for`); consecutive bands synchronize via the implicit
end-of-`omp.wsloop` barrier (`convert-par-to-omp` emits each nested wsloop
*without* `nowait`, and now recurses into the cloned `scf.for`). The shard axis
also accepts an **affine upper bound** (e.g. `N-1`), expanded to an SSA value at
emit. Measured in-house (par→omp, MATCH): **jacobi-2d 20.6×, heat-3d 19.0× @16t**
— now exceeding the affine-parallelize stand-in. lit `spmd-perband-stencil.mlir`;
ResNet-50/MNIST re-validated `0.000e+00` (counts unchanged); full lit 242/0.

**Still `par.critical` (correctly):** lu (the k-sweep body interleaves a division
glue with the update — not a clean sequential-wrapper), doitgen (shared `sum[]`
makes the outer axis genuinely non-parallel). Both are mostly-sequential anyway.

Single-thread codegen (`affine-register-block`, 2.3–2.6× over clang -O3 on
contractions, see POLYBENCH_FAMILY_FINDINGS.md) is orthogonal and composes with
this — see below.

## Codegen × parallelism compose (measured)

`affine-register-block` (register-promote the k-loop accumulator + 8×16 vector
micro-kernel) and `affine-parallelize`→omp (shard the outer `i` loop) are
**independent, multiplicative, and correct together**. Canonical GEMM N=1024,
7 iters, `mlir-runner --O3`, 16-core Zen4 (`scripts/codegen-x-parallel.sh`):

| config | median (s) | vs baseline | correct |
|---|---|---|---|
| baseline (1t)   | 2.70   | 1.0×    | — |
| **rb (1t)**     | 0.034  | **79×** | MATCH |
| par (16t)       | 0.217  | 12×     | MATCH |
| **rb + par (16t)** | 0.0026 | **1033×** | MATCH |

rb+par parallelizes the register-blocked code (one omp.wsloop over `i`, the
micro-kernel intact) → ~13× on top of rb's single-thread time. **Honest caveat:**
the 79× is over the *naive MLIR* baseline (whose k-innermost memory accumulator
`--O3` does not scalar-promote); against clang -O3 the codegen win is the
documented ~2.5×. The result here is the **composition**: both transforms fire,
the output is numerically MATCH, and parallelism multiplies the codegen speedup.
Caveat for applying to PolyBench as-emitted: register-block needs the reduction
loop **innermost** (canonical i,j,k); cgeist emits gemm as i,k,j (reduction
middle), so a reduction-innermost interchange is the precondition to compose the
two on the real kernels.

## Reproduce

```bash
# 1. generate per-kernel affine MLIR (docker cgeist), cached in bench/polybench-mlir/
scripts/polybench-gen-mlir.sh
# 2. parallel-axis opportunity map over all 30 kernels
scripts/polybench-par-survey.sh
# 3. measured OpenMP scaling + correctness for the representatives
scripts/polybench-par-bench.sh gemm 2mm syrk covariance doitgen jacobi-2d heat-3d lu deriche mvt atax
```

## Roofline (step 0) — where the walls are

Square GEMM (i,k,j), `scripts/.../roofline.sh`, Zen4 16-core; GFLOP/s = 2N³/t:

| N | working set | seq-MLIR-O3 1t | clang-O3 AOT 1t | par-omp 16t | 16t/1t |
|---|---|---|---|---|---|
| 256  | 1.5 MB | 24.7 GF | 37.2 GF | 234 GF | 9.5× |
| 512  | 6 MB   | 18.5 GF | 29.4 GF | **290 GF** | 15.7× |
| 1024 | 24 MB  | 19.5 GF | 31.5 GF | 148 GF | 7.6× |
| 2048 | 96 MB  | **9.3 GF** | **12.0 GF** | 111 GF | 12× |

Two walls confirmed: (1) single-thread MLIR-O3 is ~1.5× behind clang -O3 and both
sit at ~20–30% of peak → **codegen headroom** (register-block, #2); (2) at N=2048
(96 MB ≫ 32 MB L3) single-thread FLOP/s **halves** for *both* MLIR and clang (no
tiling) → **the L3-spill bandwidth wall → tiling (#3) pays even past clang -O3**.
16t peaks 290 GF (~29% of peak) → large compound headroom.

## Dynamic schedule for triangular bands (#1, landed)

A band whose inner extent depends on the shard IV (syrk `j:0..i`, covariance gram
triangle, trmm) is load-IMBALANCED under a static block schedule. ParBubbles now
tags such a forall `par.dynamic` (`bandImbalanced`: an inner loop bound references
the shard IV) and `convert-par-to-omp` emits `omp.wsloop schedule(dynamic)`.
Measured in-house (par→omp, MATCH): **syrk 7.4×→13.9×, covariance 7.6×→14.6× @16t**
— the triangular imbalance recovered, both now near-linear. Balanced bands are
untagged (static); ONNX bands are constant-bound so never tagged (mnist
`0.000e+00`). lit `spmd-perband-dynamic.mlir`.

## In-house par→omp coverage (landed this round)

All measured via `dr-par-bubbles{par-spmd-perband}` → `convert-par-to-omp`,
checksum-verified seq≡omp @16t: gemm 13.8×, 2mm 14.8×, **syrk 13.9×**,
**covariance 14.6×**, mvt 10.4×, **jacobi-2d 20.6×**, **heat-3d 19.0×**. lu /
doitgen stay `par.critical` (correctly: mostly-sequential / shared scratch).

## #2 register-block on the real kernels — NO-GO without size-specialization

`affine-register-block` already canonicalizes the PolyBench i-k-j GEMM order
(`canonicalizeOnce` interchanges k↔j to put the reduction innermost) **and fires
when bounds are CONSTANT** (verified: a constant-bound i-k-j GEMM → 16 `vector.fma`
micro-kernel; composes with the parallel path = the 1033× of `0ec6192`). But the
cgeist-emitted PolyBench kernels carry **symbolic** loop bounds (`ni/nj/nk`
runtime args), and the interchange guard requires constant reduction bounds —
because the downstream spatial unroll-jam/peel micro-kernel is constant-bound.
Relaxing only the guard (allow loop-invariant symbolic) lets the interchange fire
but the micro-kernel can't peel a symbolic spatial extent → it half-transforms
(unroll-jams the wrong nest, no GEMM vectorization). So **register-block does not
compose on PolyBench as-emitted**; it needs either (a) **size-specialization**
(compile for a fixed dataset → constant bounds → fires + composes, the natural
deployment path), or (b) **symbolic-bound register-blocking** (dynamic remainder
loops) — a separate, larger effort. Reverted the relaxation (added risk without
delivering); register-block lit 29/29.

## Next lever (roofline-prioritized): #3 cache-tiling within the shard

The roofline's dominant single-thread loss is the **L3 spill** (N=2048: 9.3 GF,
half of in-cache) — and it hits **clang -O3 too**, so it is the highest-value,
most broadly applicable lever, *and* unlike register-block it does **not** need
constant bounds (`dr-affine-loop-tile` is symbolic-bound-clean). Tile (i,j,k)
inside each `par.forall` shard so a thread's working set is L2-resident → cut DRAM
traffic in the bandwidth-bound regime (large-N GEMM, BLAS-2). Recommended next
step over #2.

## #3 cache-tiling within the shard — measured (tile→perband→omp)

`dr-affine-loop-tile` → `dr-par-bubbles{par-spmd-perband}` → `convert-par-to-omp`
(tile *then* shard; perband's de-affine handles the tiled imperfect nest).
Square i-k-j GEMM (the streaming/spill-prone form), `scripts/polybench-tile-bench.sh`,
checksum-verified seq≡omp:

| N | working set | seq untiled | seq **tiled** | par 16t untiled | par 16t **tiled** |
|---|---|---|---|---|---|
| 1024 | 24 MB (*fits* L3) | 17.8 GF | 11.2 GF ↓ | 146 GF | 67.6 GF ↓ |
| 2048 | 96 MB (*spills* L3) | 9.6 GF | **19.9 GF (+107%)** | 117 GF | **162 GF (+38%)** |

**Tiling recovers exactly the L3-spill loss** the roofline exposed: at N=2048
single-thread climbs 9.6→19.9 GF (back to in-cache FLOP/s) and parallel 117→162 GF
(+38%), correct. **But it HURTS when the set already fits cache** (N=1024): tiling
the i-k-j nest adds loop overhead and (for the wrong form) breaks register reuse
with no DRAM to save. So tiling is **not a free default** — it must be **gated to
the spill regime** (tile a band only when its *per-shard* working set exceeds the
effective LLC = `l3 / sharers`; the repo's contention-aware reuse-distance model
has this machinery). Caveat: for symbolic-bound PolyBench-as-emitted the size is
not known statically, so the gate needs runtime info or size-specialization.
Also note: tiling the **i-j-k** (k-innermost) form hurts (it tiles the reduction,
forcing C re-load per k-tile) — the win is specific to the i-k-j streaming form.

**Verdict:** cache-tiling within the shard is a real bandwidth-regime win
(+38% parallel / +107% single-thread at the spill point) and composes correctly
with the SPMD path.

## Working-set gate (landed) — cache-tiling as a safe default

`dr-affine-loop-tile` gained an opt-in `llc-gate=<KiB>` option (default 0 = off,
prior behavior + lit preserved): tile a band **only when its full memory footprint
exceeds the gate**, i.e. the untiled set spills the LLC and the kernel is
bandwidth-bound. **Empirically the relevant cache is the SHARED LLC, not
`l3/sharers`** — the owner-computes shards share the read-only operand (B), so the
crossover sits at the full L3 (≈32 MB) for *both* the single-thread and the
16-thread runs; an `l3/sharers` (≈4 MB) gate would over-tile and regress the
in-cache sizes. The gate threshold comes from the **MachineModel** (single source
of truth) via `llc-gate-from-model=true` (= `mm.l3Size`, built-in Zen4 32 MiB or
`cpu-cost-model-file`) — no hard-coded value; `llc-gate=<KiB>` remains as an
explicit override for tests/tuning. Gated result (square i-k-j GEMM,
`llc-gate-from-model=true`, checksum-correct):

| N | footprint | gate | par 16t untiled | par 16t **gated-tiled** | seq 1t **gated** |
|---|---|---|---|---|---|
| 1024 | 24 MB | < L3 → **skip** | 154 GF | 145 GF (≈untiled, no regression) | 18.4 GF (≈untiled) |
| 2048 | 96 MB | > L3 → **tile** | 111 GF | **162 GF (+45%)** | **19.8 GF (+113%)** |

So the gate keeps the spill-regime win (+45% parallel / +113% single-thread at
N=2048) while **eliminating the in-cache regression** (N=1024 was 67 GF / 11 GF
unconditionally tiled, now ≈ untiled). Cache-tiling is now a safe default. lit
`Analysis/DrAffineLoopTile/llc-gate.mlir`; full suite 244/0. Remaining: for
*symbolic*-bound PolyBench-as-emitted the footprint isn't known statically, so the
gate needs a runtime check or size-specialization (the constant-size path above
is the deployment case).

## Spill-points across classes — where each bandwidth wall is, and the right lever

Roofline-style scaling-vs-N (seq-1t + in-house par-16t) plus the
`dr-affine-loop-tile` decision, for one kernel per memory-behavior class
(`scripts/polybench-spill-{blas2,stencil}.sh`, Zen4, L3=32 MiB):

| class | kernel | in-cache (N=1024) | spill point | spilled | tile decision | lever |
|---|---|---|---|---|---|---|
| contraction | gemm | seq 19.5 GF | N=2048, 96 MB | seq 9.3 → gated-tile **19.9** | gate **TILE** | spatial tiling (landed) |
| BLAS-2 | mvt | seq ~1 GF (mem-bound at *all* N) | N=2048, A=32 MB=L3 | par 19.9 → **7.3** | gate SKIP / V2 **REJECT no-evicted-reuse** | **fusion** |
| stencil | jacobi-2d | seq 17.4 GF | N=2048, 2N²=64 MB | seq **6.6** (2.6× loss) | **REJECT out-of-model** | **time-tiling** |

Reading:
- **BLAS-2** (mvt/atax/gemver/gesummv/bicg) reads each matrix element ~once →
  compute intensity ≈1 flop/byte → **memory-bound at every size** (seq ~1 GF flat,
  no in-cache plateau). The parallel run still rides shared-L3 bandwidth until the
  matrix exceeds L3 (N=2048, A=32 MB), then halves. Spatial tiling is **correctly
  declined** — V2's `no-evicted-reuse` gate sees there is nothing to keep resident.
  The only cache lever is **fusion**: mvt/gemver/etc. make multiple passes over the
  same matrix, so fusing them cuts the dominant A-streaming in half
  (`dr-affine-loop-fusion`, cost-model-driven, already in-tree).
- **Stencil** (jacobi/heat/fdtd/seidel) has spatial + temporal reuse, so it runs
  compute-ish *in cache* (17.4 GF) and **spills hard at N=2048** (2N² = 64 MB > L3,
  seq 17.4 → 6.6 GF, 2.6×). Spatial tiling is declined (`out-of-model`: the
  sequential time loop wraps two spatial bands → not a tileable perfect band). The
  right lever is **time-tiling** (temporal blocking keeps a spatial tile resident
  across timesteps — `dr-affine-stencil-time-tile`, already in-tree), composed with
  the SEQWRAP parallelization.

**Net:** the spatial-tiling gate (landed) is correctly *selective* — it fires only
for the reuse-rich contraction class and stays out of BLAS-2 and stencil-spatial,
so nothing is mis-tiled. Completing cache-residency across all classes = composing
the two existing levers with the SPMD path under their own spill-gates: BLAS-2
**fusion-then-shard**, stencil **time-tile-then-SEQWRAP**.

## Composing the cache levers with SPMD — outcomes (honest)

Attempted the two remaining per-class cache levers as composes with the SPMD path.

**Stencil time-tiling — single-thread WIN, but does NOT compose with SPMD.**
`dr-affine-stencil-time-tile` (skewed time-tiling) recovers the jacobi-2D
single-thread L3 spill: N=2048, T=30, untiled seq **7.3 GF → 16.8 GF (2.29×)** at
tile-t=8/tile-s=64 (64²×2 arrays = 64 KiB → L2-resident, reused over 8 timesteps),
checksum-correct (`scripts/polybench-stencil-timetile-bench.sh`). **But** the
skewed form is a **wavefront**: the oracle classifies all three tile loops
(tt, ii, jj) SEQUENTIAL (carried) — only the innermost per-tile spatial sweep is
parallel — so `par-spmd-perband` makes the whole nest `par.critical` (0 parallel).
Time-tiling (locality) and the simple SEQWRAP (coarse parallelism) therefore
**conflict**: classic skewing serializes the tiles. Getting BOTH needs
**diamond / concurrent-start tiling** (a hyperplane-parallel schedule) — a
substantially harder polyhedral transform the skewing pass does not produce.
Net stencil picture, two regimes: **N fits L3 → SEQWRAP per-timestep = 18–20×**
(landed); **N spills L3 → time-tiling recovers single-thread 2.29×** but is
sequential; coarse parallel + locality at large N = diamond tiling (future).

**BLAS-2 fusion — NO cache lever (confirmed).** `dr-affine-loop-fusion` declines
to fuse mvt's two matvecs: they are independent (no producer-consumer) and read
the matrix in *different* patterns (A vs Aᵀ), so fusing captures no reuse and the
cost model correctly leaves them split. BLAS-2 is bandwidth-bound by its
O(N²)-work / O(N²)-data ratio (≈1 flop/byte) — there is **no spatial-cache lever**;
its only win is parallel bandwidth aggregation (have it, up to the L3 spill).

**Cache-residency status across classes:**
- contraction → spatial tiling + model-driven gate — **landed, real win** (+45% par / +113% seq at the spill).
- BLAS-2 → none (bandwidth-bound; tiling/fusion both correctly decline) — **honest no-lever**.
- stencil → time-tiling recovers single-thread (2.29×) but conflicts with SPMD; parallel+locality = diamond tiling — **partial / future**.

## #2 resolved — size-specialization is the default, not extra work

The "#2 NO-GO on symbolic bounds" turns out to be an artifact of how the survey
*extracted* kernels (as standalone functions taking `ni/nj/nk` as runtime args).
When PolyBench is compiled the normal way — whole program, fixed dataset — the
dataset dims are `#define` **compile-time constants**, so the kernel's loop bounds
are constant and `affine-register-block` fires + composes with the parallel path.

Confirmed on the actual PolyBench GEMM structure (beta-scale + i-k-j accumulate),
constant-bound N=1024, checksum-correct
(`scripts/polybench-codegen-x-parallel-pbstruct.sh`):

| config | vs baseline | note |
|---|---|---|
| baseline 1t | 1.0× | naive MLIR --O3 |
| **rb 1t** | **2.8×** | register-block (canonicalize i-k-j→i-j-k, 32 vector.fma; the *honest* codegen win, ≈ documented 2.5× vs clang -O3) |
| par 16t | 13.7× | affine-parallelize → omp |
| **rb + par 16t** | **40.0×** | both fire (parallel bands=2 after register-block), MATCH |

So per-thread codegen × parallelism (≈ 2.8× × 14×) **is** available on real PolyBench
deployment with no new pass — register-block fires on the size-specialized
(constant-bound) kernels, which is the default for a fixed-dataset compile. The
only thing that needs symbolic-bound register-blocking is the size-*parameterized*
library form (kernel as a standalone runtime-N function) — a genuinely separate,
larger effort (dynamic remainder loops), not required for the benchmark.

## #11 diamond/parallel-cache stencils — feasibility verdict (scoped, deferred)

**When it matters.** The *parallel* stencil spill point is much later than the
single-thread one, because 16 threads spread the working set across the aggregate
~64 MB L3 (2 CCDs). Measured jacobi-2D par-16t: **N=2048 (64 MB) 83.8 GF →
N=3072 (144 MB) 28.2 GF → 4096 17.2 → 6144 12.4**. So SEQWRAP-parallel stencils are
fine up to N≈2048; the parallel-cache lever only pays at **N≳3072 (grids >144 MB)**,
where it could recover par from ~28 toward the ~80 GF in-cache rate (~3×).

**Why the simple compose fails (recap).** Skewed time-tiling makes the tile loops
(tt, ii, jj) a **wavefront** (all oracle-SEQUENTIAL); only the innermost per-tile
spatial sweep is parallel. SEQWRAP wants a sequential-outer / parallel-inner-*band*
shape, so it falls to `par.critical`. Locality and coarse parallelism genuinely
conflict in the skewed form.

**Implementable designs** (both real new passes, ~1–2 days each):
1. **Overlapped (redundant-halo) row-strip tiling** — *recommended*. Shard rows
   into P strips (`par.forall`, embarrassingly parallel, no per-step barrier); move
   the time loop INSIDE each strip; expand each strip's input range by ±T (halo)
   and recompute the halo redundantly into a **private per-strip buffer**. Maps
   cleanly to `par.forall(strip){ scf.for(t){ strip±halo } }` — the inverse nesting
   of the current SEQWRAP. Cost: redundant halo compute (~halo/strip-height, e.g.
   ~30–50% at H=128/T=30) traded for cache-residency + zero sync. Needs: private
   buffer alloc per strip, halo bound arithmetic, copy-in/out, ping-pong handling.
2. **Skewed-tile wavefront parallel** — keep the existing skew, parallelize tiles
   along anti-diagonals (tiles with ii+jj=const are independent), `par.barrier`
   between diagonals. No redundancy, but needs diagonal-schedule codegen.

**Verdict:** real win (~3×) but only for very large grids (>144 MB); it is a
genuine new pass, not a session-scale change, so it is **scoped and deferred**.
Recommended approach = overlapped row-strip tiling (design above). The enabling
parallelism is sound (anti-diagonal tiles / independent strips); the work is the
halo/buffer materialization.
