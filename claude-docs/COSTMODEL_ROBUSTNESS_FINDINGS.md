# Cost-Model Robustness & Applicability — Findings

**Date:** 2026-06-08
**Author:** Claude (Opus 4.8), orchestrated (7 deep-dive agents + synthesis, ~835k tok)
**Companion:** `COSTMODEL_ROBUSTNESS_CAMPAIGN.md` (method); builds on
`COSTMODEL_REVIEW_FINDINGS.md` + `CONTENTION_AWARE_COSTMODEL.md`.

## Headline

The cost model is reliable in **exactly one regime** — Zen4 fp32/fp64 power-of-2
GEMM-family kernels on the win-path (`AffineRegisterBlock`) — and it earns its
2.3–3.4× there by **not consulting the cost model at all** (grep-confirmed: zero
references to `CacheCostModel`/`ArchHandler`/`vecBudget`). *Everywhere the cost
model actually decides* (DR keep/recompute, fission, the cost-model tiling path),
every decision reduces to a thresholded comparison against a **4-value step
latency function** — so each is a knife-edge at one tier boundary and dead-flat
between (precisely why α had to hit 100× to move anything). Three confirmed
soundness defects generate **wrong** decisions on real code; the measurement
harness that would catch them is itself unsound; and every general-applicability
extension is foundation-blocked on the still-missing per-level **bandwidth** term.

## Robustness verdict (the four dimensions, honest)

| dim | verdict |
|---|---|
| **RB1 sensitivity/soundness** | **FRAGILE, sometimes WRONG.** Step-latency → knife-edge decisions. 3 confirmed wrong-decision generators (below). `β·reg` provably **dead**. The 3 passes disagree on cache geometry for the same HW. |
| **RB2 program-shape** | mostly slow-not-wrong (over-buffer/bail), **except** two real **correctness** exposures: the distinct-SSA no-alias bet (`AffineRegisterBlock.cpp:142`, unchecked on arbitrary user code) and peel-before-`isLoopParallel` ordering. Reuse-distance is **stride-blind** (stream == gather). |
| **RB3 hardware** | **Zen4-only.** `vl=8` is an *element count* (`VectorType::get({VL},elemTy)`) → **half a zmm on fp32**; NEON-oversubscribing. Win-path has zero `ArchHandler` refs. Contention axis decorative at the default. |
| **RB4 measurement** | **UNTRUSTWORTHY as-is — and it poisons the other three.** The fp-tolerance check is a **confirmed no-op** (always PASS), `prevent_dce` disabled (DCE-to-zero structural), no pinning/turbo control. The noise "signals" we kept catching are this harness's predicted output. |

## Three confirmed wrong-decision generators (verified in code)

1. **`dr-l3-size` defaults to 0** (`Passes.td:45`) → `estimateLoadLatency`'s `l3Size>0`
   guard **skips the LLC tier** → every DR buffer in 256KB–32MB is priced straight to
   `mem=200`, and **`dr-llc-sharers` is dead on the DR path**. The "contention-aware"
   model is **inert on its own primary pass** at the default config.
2. **`kDefaultTripCount=128`** (5 sites) → a dynamic stream of 10⁶ elements is footprinted
   as 128 → under-counts reuse distance by orders of magnitude → flips keep/recompute &
   fission to the **optimistic-wrong** side. Unsafe error direction.
3. **keep/recompute prices load-*events*, not traffic** (`leafLoadCost=0` at
   `DataRecomputation.cpp:2086`, one latency per consumer, **per-consumer trip ignored**)
   → the decision is invariant to the trip count, the dimension that most determines
   whether keeping pays.

Plus: the **three passes use three cache geometries** for the same machine (DR
L2=256KB/L3=0; fission L2=1MB/L3=32MB; ARB L3=32MB), and **`vl` is element-count** so
fp32 win-path kernels silently use half the AVX-512 width.

## Top spikes (analysis → I spike; ordered, critical path first)

1. **[low] Fix the measurement floor — CRITICAL PATH.** Replace the no-op
   `polybench-verify.sh:165-172` awk with a real max-relative-error gate; add a
   **strict-IEEE** (`-fno-fast-math`, no-MLIR) reference oracle; make `t<=0`/NaN a hard
   FAIL (not a dropped/"fast" win). *Self-validating* (feed 1e6× error → must FAIL; 1e-12
   → PASS). **Also answers the codegen campaign's biggest open question: are the
   FMA-contract / dot-reduce / peel reassociations numerically sound?** (never checked vs
   strict IEEE). Every downstream runtime spike needs this.
2. **[low] Fix `dr-l3-size=0` + unify cache geometry.** Set drL3Size=32MB/drL2Size=1MB
   (one shared `CacheParams` default). IR-diff (noise-free): a 512KB buffer flips
   RECOMPUTE→KEEP and `dr-llc-sharers∈{1,8}` now moves a decision (it provably can't
   today). Histogram PolyBench buffer sizes to size the impact. Revives the contention
   work on the DR path.
3. **[med] Stride-aware footprint** — lift the *existing* `estimateAccessLatency`
   line-granular logic (`RematKernel.cpp:390`, today quarantined to partial-remat) into
   `estimateOpFootprintBytes`. Build-once-propagate (DR + fission + tiling all consume it).
   IR-diff: a gather flips to recompute/skip while a stream holds. **Must first measure
   `estimateAccessStrideElements` coverage** (nullopt-bail rate) or it's inert.
4. **[med] Win-path arch-portability** — derive `vl=vectorWidthBits/elemBits`, `mr` from
   `vecBudget` (wire the existing `ArchHandler`). Fixes the fp32 half-zmm bug; tests
   whether "register-FIT not load-bearing / vl=16<vl=8" survives a forced 16-reg budget.
5. **[high] Unified 3-arm `decideRematStrategy`** (recompute | materialize+reload | fuse)
   on one currency `N·estimateLoadLatency(distance)` — the gemver over-rejection root
   cause; white-space vs production. (FUSE *scores* but DR/fission can't *execute* it —
   the affine fork can; partly aspirational.)

## Dead ends (honest)
- Conflict-miss/padding cost model — schema-blocked (no associativity in CacheParams) AND
  Zen4's 16-way 96MB V-cache likely makes conflicts a non-problem; do a cheap perf-counter
  pre-check, expect <5% → close.
- SVE/scalable-vector handler — not robustly spikeable on the x86 box (QEMU timing
  untrustworthy); the AVX-512 masked-tail sub-part folds into #4.
- Full barvinok parametric footprint — research-grade; do the cheap conservative-streaming
  sentinel first, gate barvinok on a count showing the 128-fallback flips many decisions.
- Standalone `kIssueWidth` parameterization — off the win-path; fold the FMA-floor into #4.
- **Re-running (α,β,γ) calibration before fixing the harness** — that finding is plausibly
  itself a noise artifact of the unpinned/median-of-3 objective; fix #1 then re-confirm.

## Cross-cutting
- **Foundation gap:** the model ranks cache *tiers* but cannot form *bytes/cycle* → can't
  soundly price traffic-vs-recompute or roofline regime shifts. The per-level **bandwidth
  term** gates AP1/AP3 magnitude-correctness; flagged, not assumed.
- **The contention axis is simultaneously the claimed novelty AND "decorative at the
  default" (L3=0 + sharers=1).** Spike #2 resolves this — it's only novelty once revived.
- **Prefetch caveat (recurring):** Zen4 line-fill prefetch + the 96MB V-cache may hide
  stream-vs-gather and over-fusion — the same uarch reality that refuted register-FIT.
  Spikes #3 and the AP1 fuse-arm carry this as their explicit falsifier.

## Findings docs cross-ref
Full per-track findings + gap table in the workflow result; this is the actionable digest.
