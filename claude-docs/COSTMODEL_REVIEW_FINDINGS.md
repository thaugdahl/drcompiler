# Cost-Model Review — Findings

**Date:** 2026-06-08
**Author:** Claude (Opus 4.8), orchestrated (4 deep-dive track agents + synthesis, ~430k tok)
**Companion:** `COSTMODEL_REVIEW_CAMPAIGN.md` (method, scope, surface inventory)
**Supersedes:** `COSTMODEL_FINDINGS_CLAUDE.md` (2026-06-03), `COSTMODEL_SPIKE_FINDINGS.md`
(2026-06-04) — both disregarded per user; this is the fresh full-surface analysis.

---

## Headline (the one fact that reorganizes the review)

**`AffineRegisterBlock.cpp` — the path that produces every headline 2.3–3.4× win —
references the analytical cost model ZERO times.** Grep-verified: no `ArchHandler`,
`CacheParams`, `combineCosts`, `estimateLoadLatency`, `vecBudget`, `classifyType`,
`estimateComputeCost`. `mr=8`, `nr=16`, `mc/nc/kc=256`, `vl=8` are hardcoded knobs; the
only "cost" decisions are structural legality predicates plus two crude guards (the
`worthTiling` extent>tile test and the `rank>=3` vectorize gate at `:835`).

Consequence: **the elaborate cache cost model (`CacheCostModel`/`ArchHandler`) and the
codegen wins live in two disjoint universes.** Refining the DR/fission cost model (latency
curve, footprint, roofline currency) does **not** touch the kernels the project is known
for. The highest-value work is building the **first analytic cost term the win-path ever
consults** — and that term is the **explicit-vector micro-kernel cost**, where four
independent findings (T1-2 kernel cost, T1-4 register-FIT, C4-1 exact-accumulator
residency, the latency-hiding floor) collapse into **one BLIS-shaped model**. It is also
the **one place register pressure is analytically trustworthy**: the explicit kernel pins
exactly `mr·⌈nr/vl⌉` vector `iter_args` that OoO rename cannot un-allocate — unlike scalar
spill, which the prior Zen4 spike showed is fiction on big-OoO x86.

---

## The disjoint-universes map

| World | Drives | Cost basis | Touches the wins? |
|---|---|---|---|
| `AffineRegisterBlock` | register-block / cache-tile / vectorize | **none** — hardcoded knobs + structural guards | **YES** (2.3–3.4×) |
| `CacheCostModel`+`ArchHandler` | DR keep/recompute, BufferElim | `α·mem+β·reg+γ·alu` linear sum, 4-bracket latency step | no |
| `MemoryFission` | loop fission | `keep = compute+1+nCons·L1lat` (a 4th, inconsistent notion) | no |
| `LoopFusion`/`LoopTiling` forks | fuse/tile | same linear `combine()`, `liveVals=c²` spill term | no |

Three of four cost notions never reach a headline kernel. The review's leverage is
therefore (a) give the win-path its first real cost term (the BLIS-style vector model),
and (b) fix the one cache-model defect that flips a *real* decision (the MemoryFission L1
fiction — the user's own gemver over-rejection).

---

## Ranked ideas (impact × tractability, double-counts merged)

| # | Idea | Track | Impact | Tract. | Note |
|---|---|---|---|---|---|
| 1 | **SLP-failure predictor** replaces the rank gate | T1 | high | high | gateway; reclaims 2D strided/gather SLP loses |
| 2 | **Explicit-kernel cost + BLIS register-FIT + latency floor** selects (mr,nr,vl) | T1+C4 | high | med | merged T1-2/T1-4/C4-1; BLIS parity on the exact-count path |
| 3 | **Kill MemoryFission L1 fiction** (size-aware tier) | T3=T2 | high | high | the one cache idea that flips a real decision |
| 4 | **vl=16 width test** — beat vs only confirm the rank gate | T1 | med | med | empirical disambiguator; honest likely-negative |
| 5 | **Scope SLP/scalar register term to inert** | C4 | med | high | honest cleanup; the justification for ranks 1–2's reg use |
| 6 | Analytical working-set test replaces `worthTiling` | T2 | med | med | Polly already ships this; reaches parity |
| 7 | Roofline `max(compute,mem)` currency | T2 | high | **low** | north-star; **blocked**: ArchParams has no bandwidth field |
| 8 | Smooth associativity-aware latency curve | T3 | med | med | off hot path; payoff unproven until edge-histogram measured |
| 9 | Cache-line / stride-aware reuse-distance footprint | T3 | med | med | behind BLIS+TTI; coverage risk if stride bails |
| 10 | Remove `LoopTiling` anti-predictive `c²` spill term | C4 | low | med | category error (reg spill on cache-tile loop) |
| 11 | Arithmetic-intensity ridge gate (fire/decline) | T2 | low | med | mostly re-derives existing declines; needs ArchParams ridge |
| 12 | Roofline FMA-throughput compute floor | T3 | low | med | off hot path (no `estimateComputeCost` call) |

---

## Top spikes (the actionable core — "analysis, then I spike")

### Spike 1 — SLP-failure predictor replaces the rank gate  *(low effort, gateway)*
- **Hypothesis:** the explicit vector kernel beats LLVM-SLP on a 2D accumulator whenever
  SLP must gather or scalar-tail — column-major B, non-`nr`-divisible trip, mixed type —
  exactly the cases `rank>=3` ships to SLP and loses.
- **Build:** debug flag bypassing `:835`; compute `slpLikelyOK = (every multiplicand
  innermostStrideOne-in-j OR j-invariant) ∧ accRank≤2 ∧ single-type ∧ const trip,
  trip%nr==0 ∧ trip≥nr`; fire explicit when `!slpLikelyOK`. Reuses `innermostStrideOne`
  (:690), `canVectorizeDAG` (:741), `getConstantTripCount`. No new analysis.
- **Measure:** 4 micro-kernels on 7950X3D core 8, median 11, checksum: (a) contiguous-B
  [baseline], (b) column-major B, (c) N=257 tail, (d) mixed-type. SLP vs forced-explicit.
- **Signal:** explicit wins (b)/(c) by >15%, ties (a). **Falsifier:** explicit also loses
  (b) → LLVM 22 strided-loads column-major B fine on x86 now (post PR#80310 generalizing
  off RISC-V); narrow the predictor to tail/type clauses and report the gather clause obsolete.

### Spike 2 — Explicit-kernel cost + register-FIT + latency floor select (mr,nr,vl)  *(med)*
*(merged T1-2 / T1-4 / C4-1 / T2-5 — four framings of one BLIS inequality)*
- **Hypothesis:** on the explicit path live-vector count is an EXACT invariant, so BLIS
  FIT (`accVecs+operandVecs ≤ Nvecreg`) and the latency-hiding floor (`accVecs ≥
  L_fma·T_fma ≈ 10` on Zen4) predict the GFLOP/s ridge. Hardcoded 8/16/8 happens to
  satisfy both on fp32/AVX-512 but is provably wrong on fp64/AVX2 (8×16/vl=4 = 32 vregs >
  16 ymm).
- **Build:** `costVectorKernel(band,vl)` → `accVecs = mr·⌈nr/vl⌉`, per-k issue = 1 load + s
  broadcasts + accVecs FMAs; `liveVecs = accVecs + ⌈nr/vl⌉ + (s>0?1:0)`. Use `classifyType`
  for exact regs/accumulator. Pick largest tile with `liveVecs≤vecBudget ∧ accVecs≥floor`,
  replacing 8/16/8 and the comment-only `(maxMul>2)?2u:4u` dot shrink (:945). Behind a flag.
- **Measure:** sweep mr∈{4,6,8,10,12} × vl∈{4,8,16} for gemm + bmm/ttm; overlay FIT line +
  floor; force vl=4 to provoke fp64-style over-allocation, capture `mem_uops_retired`.
- **Signal:** ridge at largest tile under FIT, above floor; forced over-FIT shows a spill
  cliff. **Falsifier:** peak at a FIT-violating tile with no slowdown → rename absorbs even
  pinned vectors → downgrade to floor-only (keep floor, drop ceiling). C4's own kill condition.

### Spike 3 — Kill the MemoryFission L1 fiction  *(low effort, real decision flip)*
- **Hypothesis:** fission over-fires because every materialized load is priced at
  `l1Latency` (4cy) while the buffer is sized by the trip count — so any buffer worth
  fissioning does NOT fit L1. The pass already ships `estimateLoadLatency` but doesn't call it.
- **Build:** delete `:321 loadLatency=l1Latency`; `bufferBytes = elemBytes·trip` (trip
  already extracted at :402-419; pessimistic streaming fallback when dynamic); call
  `dr::estimateLoadLatency(bufferBytes, cache)`; price 2nd+ consumer cold when buffer
  exceeds the level. One call site.
- **Measure:** gemver-style shared subexpr at 3 sizes (L1 ~2K, L2 ~30K, L3+ ~5M f64);
  fission ON/OFF, wall-time.
- **Signal:** current fires all 3; size-aware keeps L1, flips OFF at L3, matching the faster
  binary. **Falsifier:** recompute so dear (transcendental) that keeping a DRAM buffer still
  wins → correctness-cosmetic only.

### Spike 4 — vl=16 width: beat the rank gate, or only confirm it?  *(med, honest-negative likely)*
- **Hypothesis:** "SLP wins 2D" is a WIDTH effect (SLP nr=16 vs explicit vl=8) recoverable
  by vl=16; OR a register-allocation effect (SLP's 8×2 grid allocates better than 8×1
  vl=16) in which case the model can only cheaply confirm the gate.
- **Build:** reuse spike-1 bypass + spike-2 selection. Configs on contiguous gemm: (A) SLP
  nr=16, (B) explicit vl=8 [today's loser], (C) explicit vl=16.
- **Measure:** N∈{512,1024,2048}. **Signal:** C≈A (±3%) ≫ B → retire the gate. **Falsifier:**
  A beats C by >5% → at vl=16 accVecs=8 < ~10 floor, SLP genuinely near-optimal → keep the
  gate, downgrade model to confirmation. The floor term *predicts* this is the likely outcome
  — report it honestly, it is a real negative, not spin.

### Spike 5 — Scope the SLP/scalar register term to inert  *(low effort, cleanup)*
- **Build:** gate `analyzeHypotheticalStatic` penalties in BufferElim/Fusion to vector-typed
  closures only (exact count); scalar → 0. Leave DR `:2093-94` at 0 but re-annotate as
  DELIBERATE inert (not stale TODO). Evaluate deleting `LoopTiling:311 liveVals=c²` (C4-2).
- **Measure:** differential over PolyBench, penalty ON vs forced-0-on-scalar.
- **Signal:** few decisions flip, no regression → inert fiction, scoping is free correctness.
  **Falsifier:** a regression → the term carries signal → keep+validate (don't assume).

---

## Consolidated gap-vs-SOTA table

| Idea | Analytical (BLIS / linalg) | Polyhedral / Roofline | Production (LLVM/GCC/Polly) | Net standing |
|---|---|---|---|---|
| **SLP-failure predictor** | BEHIND: BLIS never bets on a downstream auto-vectorizer; emits the µkernel directly. Idea closes the implicit bet. | Stride clause IS a roofline-regime test (col-major B drops AI ~nr/2→1, memory-bound). | Reconstructs TTI `getGatherScatterOpCost` from the affine map BEFORE info is lost — strictly more semantic info than post-lowering TTI. | closes gap; ahead of TTI on info |
| **Kernel cost + FIT + floor (merged)** | AT PARITY with BLIS core (`mr·nr+nr+1≤Nreg` + `mr·nr·lanes≥L·T`). `classifyType` already gives exact regs/value. Currently unused. | Register file = innermost roofline tier; FIT sets max in-register AI; the ~10-accum latency floor is the single missing formula. Pluto/Polly punt register sizing to backend. | Proactive sizing vs LLVM/GCC REACTIVE spill-after-the-fact — THE BLIS advantage; on rank≥3 LLVM's model isn't even engaged. | parity w/ BLIS, ahead of production on rank≥3 |
| **MemoryFission size-aware tier** | trivially behind any analytical model; fix is self-consistency. | roofline crossover: DRAM stream buffer converts compute-bound recompute→memory-bound load. | more aggressive than any production fission (LLVM LoopDistribute is dependence+cost-guarded). | closes a self-inflicted inconsistency |
| **vl=16 width** | BLIS sets width=vl, expresses nr as vector columns; our nr/vl conflation is behind. | width drives the latency floor: vl=16→8 accum, borderline vs ~10 — the concrete reason it may NOT pay. | production gets 16-wide via VF×UF (SLP implicitly UF=2). | parity at best; may confirm gate near-optimal |
| **Scope register term** | BLIS reasons about registers ONLY for the µkernel — scoping matches BLIS discipline. | Pluto/roofline carry NO register term at loop-transform level. | LLVM/GCC apply NO register penalty to high-level fusion/remat. | scoping = SOTA-aligned |
| **Working-set tile test** | this IS the BLIS contribution (mc/kc/nc from cache geometry). | reuse-distance gives same answer; Pluto tiles but doesn't pick sizes from capacity. | **Polly already ships it** (`getMicro/MacroKernelParams`) — we are BEHIND Polly. | behind Polly; idea reaches parity |
| **Roofline currency** | BLIS is implicitly `max(compute,mem)`; linalg/TTI ADDITIVE (can't express "this load is free"). | IS the roofline model as arbitration currency; Pluto never forms bytes/FLOP. | LLVM/GCC/ICX additive reciprocal-throughput; none use roofline max(). **BLOCKED: no bandwidth field in ArchParams.** | north-star; blocked on schema, off hot path |
| **Smooth latency curve** | BLIS treats cache graded+associativity-aware; TTI `getMemoryOpCost` is a per-op constant. | Gysi-Grosser HayStack (PLDI'19) / AET-MRC compute continuous miss-ratios; our sigmoid is an O(1) surrogate. | TTI/LoopVectorize model NO size-dependent latency. | ahead of TTI on latency axis; payoff unproven |
| **Stride-aware footprint** | BLIS counts cache lines natively; TTI `getGatherScatterOpCost` distinguishes stream/gather — we are BEHIND. | core of polyhedral footprint (isl/barvinok distinct-line counting; cache-oblivious Q(n;Z,L) line-granular). | TTI separates MemoryOp/GatherScatter/Interleaved; GCC has aligned/unaligned/gather variants. | behind both; idea reaches parity |

---

## Dead ends (do NOT pursue — honest, with reason)

- **Unified CodegenCandidate scorer (T2-4):** downstream of a roofline currency that
  **cannot be computed yet** (ArchParams has no bandwidth/peak-FLOP field). Building an
  oracle on top of the linear sum it is meant to replace. Defer until the currency lands
  AND a cache/fission decision is shown to flip.
- **Sequence-search arbitration (TVM/Ansor):** transforms compose in a dependence-FIXED
  order — this is the **BLIS regime** (analytical scoring of ~5-10 configs), not the Ansor
  regime. Evolutionary search is unnecessary and infeasible on one serial box.
- **Operand packing:** flagged a RED HERRING in memory (`codegen_campaign_phase0`). No
  track resurrected it; noted so it is not reintroduced under the BLIS-parity banner.
- **Resurrecting β·reg as a general always-on term on the SLP/scalar path:** unfalsified
  (DR hard-zeros it, `:2093-94`), never validated; ~192 physical regs + rename make
  architectural-fit fiction on big-OoO x86. Keep inert/scoped; don't wire live without the
  validation that was never done.
- **Roofline compute floor (T3-4) as a near-term spike:** off the hot path (no
  `estimateComputeCost` call from the win-path). Don't burn serial bench time.
- **Per-machine Nelder-Mead calibration:** premature — calibration only pays once the model
  *shape* is right (roofline currency, smoothed curve). Sequence after, not before.

---

## Empirical results (spike phase, 2026-06-08, Ryzen 7950X3D core 8, checksum-verified)

The spikes **overturned the synthesis's top two register ideas by measurement** and
surfaced a lever neither the agents nor I anticipated. Honest, spike-first.

### Spike A — mr × vl sweep on the explicit path (bmm/ttm, rank-3) → register-FIT REFUTED

Sweep of `mr ∈ {4,8,12,16,24,32} × vl ∈ {4,8,16}` via CLI options (no rebuild), bmm & ttm
N=128 cubic, all checksums MATCH. Representative (bmm, vl=8): mr=4→49.0, **mr=8→50.6**,
mr=12→**21.0**, mr=16→55.0, mr=24→**21.9**, mr=32→49.6. ttm vl=8: mr=8→66.7, mr=12→**23.6**,
mr=16→66.9, mr=24→**23.5**.

- **The BLIS register-FIT ceiling is NOT load-bearing.** `accVecs=mr` (the broadcast
  explicit kernel keeps j as a vl-loop — it does NOT unroll-jam by nr, verified at the call
  site), so at vl=8 the live-vector count `mr` is far under the 32-zmm file for all tested
  mr. The FIT ceiling never binds; the latency floor (mr≥8) is a **soft ~10%** (mr=4 only
  trails mr=8 by ~10%), not a cliff. **The prior spike's "fixed tile, no cost model"
  verdict extends to the explicit tensor path.** Spike-2 (FIT-based selector) is dead.
- **vl=16 is consistently WORSE than vl=8** (bmm mr=8: vl=8→50.6 vs vl=16→38.3). **Spike-4
  / idea T1-3 ("raise vl to 16") is REFUTED** for the broadcast kernel — wider accumulators
  halve the independent j-blocks in flight and the kernel is load/broadcast-bound, not
  width-bound.
- **The real lever is `mr | outer-trip` divisibility.** mr=12,24 (∤128) collapse 2.5×;
  mr=4,8,16,32 (|128) are fast. Mechanism verified: mr=12 emits 9 `affine.for` vs mr=8's 5
  — a **scalar remainder loop** (128=12·10+8), no register spill. By Amdahl, ~6% of work at
  naive speed costs ~2.5×.

### Spike B — the inner (vl) divisibility cliff is a GENERALIZATION GAP

bmm, mr=4 fixed (divides every N below), vl=8, varying whether `vl | N` (inner j trip):

| N | vl\|N? | explicit fires? (`affine.vector_load`) | rb GFLOPs | clang | **rb/clang** |
|---|---|---|---|---|---|
| 96  | yes | 8 (fires) | 65.9 | 16.5 | **3.99×** |
| 104 | yes | 8 (fires) | 59.1 | 18.6 | **3.17×** |
| 120 | yes | 8 (fires) | 57.1 | 18.8 | **3.04×** |
| 128 | yes | 8 (fires) | 45.9 | 19.4 | **2.37×** |
| **100** | **no** | **0 (bails→SLP)** | **15.2** | 18.4 | **0.83× (LOSS)** |
| **124** | **no** | **0 (bails→SLP)** | **14.2** | 18.7 | **0.76× (LOSS)** |

`vectorizeBroadcastBand` bails at `:825` (`trip%VL!=0 → failure`) whenever the inner trip
is not a multiple of vl, dropping the band to the SLP fallback — which is **worse than
clang** on those sizes (clang masks/peels the tail; our pass does not). **So on arbitrary
problem sizes register-blocking is a net LOSS.** Every headline win (N=128/1024/2048) is
vl-divisible by luck of the benchmark sizes. This is the size-fragility the generalization
thesis must close.

### Reframed conclusion — the highest-value improvement is REMAINDER/TAIL handling, not a register cost model

The measured levers are **trip-count divisibility**, both inner (`vl | j`, a hard bail to a
losing fallback) and outer (`mr | i`, a scalar cleanup loss) — exactly what a cost-aware
codegen pass should handle, and exactly what the agents' register-FIT/vl-16 ideas do *not*
address. The validated top improvement is now:

> **Peel/mask the vl-remainder** (vectorize the `⌊j/vl⌋·vl` main part, handle the tail with
> a masked vector or scalar peel) so the explicit kernel fires on **any** N — converting the
> 0.76× loss into a ~3× win on awkward sizes. Secondarily, register-block the mr-remainder
> instead of leaving it scalar. This is idea T1's `:824-826` weakness made actionable, and
> it is the single change that makes the wins **size-robust** (true generalization).

Gap vs SOTA: this is exactly what production vectorizers (LLVM LoopVectorize, GCC) and BLIS
do — masked/peeled tails are standard; our hard-bail is the outlier. We are **behind
production** here, and closing it is parity, not novelty — but it is the difference between
"wins on power-of-2 demos" and "wins on real sizes."

**Revised spike ledger:** ❌ register-FIT (Spike 2, refuted) · ❌ vl=16 (Spike 4, refuted)
· ✅ **vl-tail peel** (new #1, validated cliff, implementable) · ⬜ MemoryFission L1 fiction
(Spike 3, pending) · ⬜ SLP-failure predictor (Spike 1, lower urgency — the "when does
explicit beat SLP" question is answered by divisibility, and the pressing issue is that it
*bails entirely* on tails).

### vl-tail peel — IMPLEMENTED + VALIDATED (`AffineRegisterBlock.cpp` `vectorizeBroadcastBand`)

When the inner spatial trip isn't a multiple of VL, instead of bailing to SLP, split `sIn`
into a vl-divisible **main** loop (vectorized) + a **scalar tail clone** (`[VL·⌊trip/VL⌋,
ub)`, left memory-backed; LLVM compiles it). Placed **after** the rank/stride/DAG checks so
only bands that will actually vectorize are split (a rank<3 band bails first — that ordering
bug cost one lit failure, now fixed); bails when `trip < VL` (empty main).

Measured (bmm, mr=4, vary `vl | N`):

| N | vl\|N | before (bail→SLP) | **after (peel)** | clang |
|---|---|---|---|---|
| 96  | yes | 3.99× | 3.80× | (unchanged, peel inactive) |
| **100** | no | **0.83× LOSS** | **2.01× WIN** | — |
| **124** | no | **0.76× LOSS** | **2.02× WIN** | — |
| 128 | yes | 2.37× | 2.54× | (unchanged) |

- **Net: a net loss on non-vl-divisible N became a 2× win** — register-blocking on the
  explicit (tensor) path is now **size-robust**. Recovery is 2× not the ~3–4× of aligned
  sizes because the scalar tail (≤VL cols) runs slow (Amdahl); a **masked-vector tail**
  would close the rest (future refinement).
- **No regression:** lit **9/9**; full 13-kernel suite all **MATCH** at N=1024/128 (peel
  inactive at divisible N — identical behavior: gemm 2.38, syrk 2.68, bmm 2.70, ttm 3.31).
- **Multi-matmul double-processing risk = moot.** The explicit-path kernels (bmm/ttm/tcon2)
  are single-reduction; rank-2 multi-matmuls (2mm/3mm) bail at the rank gate to SLP and
  never reach the peel (verified: `affine.vector_load`=0 for 2mm even at N=128).

### Related finding — the 2D (SLP) path is ALSO size-fragile (separate, unfixed)

The peel fixes the **explicit (rank≥3)** path only. Rank-2 kernels (gemm/2mm/3mm/covar/trmm)
bail at the rank gate to the **SLP path**, whose own `mr×nr` unroll-jam leaves scalar
remainder loops at awkward N. Measured (unaffected by the peel — pre-existing): **2mm N=100
→ 0.74×, 3mm N=100 → 0.61×** (both MATCH; same kernels win 2.4× at N=1024). So the
divisibility cliff hits *both* paths; only the explicit one is now handled. Closing the 2D
side needs an analogous remainder strategy on the scalar-promote path (or deferring
small/awkward-N nests to clang) — the natural follow-on, and a real caveat to the
"generalizes across BLAS-3" claim: **it generalizes at vl/mr-divisible sizes; awkward sizes
need tail handling.** (A clean 2D divisible-vs-awkward small-N sweep was blocked by a
`gemm128` driver `-DN` param bug — the clang reference itself went NA at N≠128 — so the 2D
cliff is shown via 2mm/3mm, not a gemm sweep.)

---

## Spike 3 — MemoryFission: the cache-fit cost model is unsound under contention (agents' fix REFUTED)

The agents (and I, initially) proposed making fission size-aware: skip when the materialized
buffer exceeds the cache. Spike-first measurement **overturned this** — twice over.

**Step 1 — a noise trap (process honesty).** A first run at M=1e6/8MB, single kernel call
(~2ms), showed fission 1.31× *faster* and I wrongly concluded the over-firing claim was
false. The user caught it: **2ms is within noise.** Re-measured with a 61MB buffer, an inner
repeat loop (~0.4–0.9s/sample), warmup discarded, and reported spread (1–8%) — the sign
**flipped**: fission *hurts* 0.41–0.35× at 61MB. Lesson logged: never report a verdict from a
sub-noise measurement.

**Step 2 — isolated crossover (robust):**

| buffer | on/off | |
|---|---|---|
| 1–7MB | 1.35–1.87× | HELP |
| 15–30MB | 1.05–1.07× | help (marginal) |
| 61MB | **0.41×** | **HURT** |

Knee at **L3 (32MB)** — fission helps while LLC-resident, hurts once it streams from DRAM.
The current model (`loadLatency=l1Latency` unconditional, `MemoryFission.cpp:321`) says
FISSION at every size → **over-fires above L3, confirmed.** So far the agents were right.

**Step 3 — L3 contention demolishes the cache-fit premise (the real finding).** L3 is shared
(7950X3D CCD1: 32MB across cores 8–15). Re-ran isolated vs 7 stressors on cores 9–15
(same binaries):

| buffer | ISOLATED | **CONTENDED** |
|---|---|---|
| 0.23MB | 1.93× | **0.72× HURT** |
| 0.9MB | 1.89× | **0.75× HURT** |
| 4MB | 1.80× | **0.71× HURT** |
| 15MB | 1.08× | **0.62× HURT** |

This first-pass table (buffer alloc *inside* the timed loop) showed fission hurting at every
size, even 0.23MB. That conclusion was **partly confounded** — see Step 4.

**Step 4 — the confound, and the corrected (contention-aware) picture.** Two bugs in Step 3:
(a) a 240KB buffer exceeds glibc's 128KB `M_MMAP_THRESHOLD`, so each per-call `memref.alloc`
`mmap`'d cold-faulted pages — a measurement artifact, not fission's cost; (b) it gated on the
buffer alone, not the **total** working set (x + buf + 3 outputs). Re-measured with warm
allocation (`MALLOC_MMAP_THRESHOLD_` high) and the working set sized against the **private
1MB L2** (per-core, contention-immune — *not* the shared L3):

| total WS | buffer | ISOLATED | **CONTENDED** |
|---|---|---|---|
| 200–400K | 40–80K | 2.0× | **2.0× HELP** |
| 800K | 160K | 1.24× | 0.47× HURT |
| 1.6M | 320K | 0.87× | 0.42× HURT |
| 4.6M | 0.9M | 0.72× | **2.58× HELP** |

**Corrected verdict (non-monotonic; the contention-aware model):**
- **Private-L2-resident fission is contention-immune.** When the *whole* working set fits the
  per-core L2 with margin (≤400K ≪ 1MB), fission helps 2.0× loaded = isolated. So a cache gate
  **is** sound — but it must be the **private L2** (the cache you can count on), not the shared
  L3 (which a co-tenant can evict). This is the central contention-aware principle: *model the
  cache you're guaranteed, not the cache you nominally have.*
- **The mid regime (WS ≈ L2 … few×L3) is non-monotonic and genuinely contention-dependent.**
  WS 800K–1.6M: the extra buffer tips the set over private L2 → spills to shared L3 → hurts
  under load. WS 4.6M: fission *helps 2.58× under load* because it reads the source **once**
  and the freshly-produced buffer stays warmer than the source that recompute re-reads 3× (the
  stressors evict the stale source faster than the just-written buffer). So at large WS fission
  *protects against* contention by cutting source re-reads — the original fission rationale,
  re-appearing.
- **The L1-fiction over-fires** for the WS≈L2…L3 band (says FISSION where it hurts under load),
  and the agents' physical-L3 gate is still wrong (shared, evictable). The right model uses the
  **effective** cache: private L2 as the guaranteed budget, shared L3 derated by expected
  sharers, and a worst-case bandwidth for the re-read/round-trip traffic.

**Meta:** the Step-3→Step-4 reversal is itself the lesson — even a "robust" measurement
(spread 1–8%) encoded a hidden assumption (warm alloc, exclusive cache). Three agent
recommendations were overturned by measurement (register-FIT, vl=16, fission-physical-L3), and
one of *my own* over-corrections (fission-never-helps-under-contention) was overturned by the
user's L2 question. The throughline for the contribution: **a contention-aware cost model keyed
on guaranteed (private) cache + worst-case bandwidth**, not exclusive-cache analytical models
(BLIS/Pluto/roofline all assume the cache is yours).

## Cross-cutting open questions (carried into the spikes)

1. Does LLVM 22 now strided-load column-major B on x86 well enough that the predictor's
   gather clause has weakened? (Spike 1 resolves.)
2. Is "SLP wins 2D" a WIDTH effect (vl=16 recovers it) or a register-allocation effect
   (gate is near-optimal)? Opposite implications for whether the model can beat or only
   confirm. (Spike 4 resolves; floor term says likely the latter.)
3. Does the exact-accumulator FIT bound actually shape the GFLOP/s ridge on Zen4, or does
   rename flatten it like it flattened the SLP-path architectural fit? (Spike 2's kill
   condition.)
4. ArchParams bandwidth/peak-FLOP schema: per-level BW or a single DRAM-BW ridge for the
   first cut? (Gates ideas 7/11/12.)
