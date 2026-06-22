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
with the SPMD path; productionizing = wiring a per-shard working-set-vs-LLC gate
into the tile→shard pipeline so it fires only where it pays.
