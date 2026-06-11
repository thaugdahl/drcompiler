# Cost-Model Robustness & General-Applicability Campaign

**Date:** 2026-06-08
**Author:** Claude (Opus 4.8), orchestrated for Tor
**Status:** ACTIVE — survey phase
**Builds on:** `COSTMODEL_REVIEW_FINDINGS.md`, `CONTENTION_AWARE_COSTMODEL.md`
(the contention-aware reuse-distance model + explicit vectorization landed this
session).

## Goal

Two questions, now that the contention-aware reuse-distance cost model is in place:

1. **Robustness** — where are the model's decisions *reliable* vs *fragile or
   wrong*? Sensitivity to uncertain inputs, the shapes of real (non-GEMM) code,
   hardware/contention variation, and whether the *evidence* the model rests on
   is sound.
2. **Applicability** — what *general memory-facing optimizations* could the
   contention-aware reuse-distance model soundly drive, beyond register-block /
   fission / recompute?

Per-idea deliverable (as before): **finding/weakness → improvement or extension →
mini gap-analysis vs SOTA** (analytical: BLIS/MLIR-TTI; polyhedral+roofline:
Pluto/reuse-distance/cache-oblivious; production: LLVM/GCC/Polly). Honest,
spike-first, no spin. Output: analysis → I spike the top 1-2.

## The model as it stands (what the agents analyze)

- **`CacheCostModel`** (DataRecomputation): reuse distance = `estimateInterveningFootprint`;
  `estimateLoadLatency` now derates the **shared** L3 by `llcSharers` (private L1/L2
  full); `decideBufferStrategy` (keep vs recompute), `decideBufferElimination`.
- **`MemoryFission`**: FISSION iff `totalWS ≤ l2OccupancyPct·privateL2` OR
  `perConsumerReuseDist > l3Size/llcSharers`. Dynamic trip → legacy fallback.
- **`AffineRegisterBlock`**: explicit vectorization for all ranks (broadcast vector
  µkernel + dot reduction-vec, no SLP reliance — guarded fallback only; FMA via
  `fastmath<contract|fast>`); cache-tile hook: tile when `WS > l3Size/llcSharers`,
  tiles clamped to fit.
- **`ArchHandler`**: `combineCosts = α·mem + β·reg + γ·alu` — found **not locally
  calibratable** (decisions flip only at 30–100× weights; `β·reg` inert/scoped).

## Established findings the agents must not re-litigate

register-FIT not load-bearing on the explicit broadcast kernel (holds `mr` vectors,
not `mr·⌈nr/vl⌉`); vl=16 < vl=8; **FMA was missing** on the broadcast family (big
win once `contract` set); `l2OccupancyPct=50%` validated robustly; combiner weights
not locally calibratable; SLP reliance removed. Methodology scars: runtime Δ <~1% is
noise (use **repeats** to separate signal from noise; **IR-diff** for noise-free
decision checks; **checksum-MATCH** not just valid IR); cgeist FAILs (2mm/3mm →
0.000000) bias geomeans.

## Tracks (survey, parallel — 1 deep-dive agent each + synthesis)

**Robustness**
- **RB1 Decision sensitivity & soundness** — how far are decision boundaries from
  the operating point (cf. α flips at 30–100×); sensitivity to `kDefaultTripCount=128`,
  buffer-size/trip estimates, `llcSharers`, element types; unfalsifiable terms;
  wrong-decision / FAIL / noise-fit modes.
- **RB2 Program-shape robustness** — where static estimates break: dynamic shapes,
  gather/irregular/strided access, aliasing, imperfect nests, mixed/odd types,
  non-affine. The "real code isn't a clean GEMM" axis.
- **RB3 Hardware/contention robustness** — microarch portability (tile, vl, FMA,
  reg file, the "8×16 within 5%" claim off-Zen4); validity of `l3Size/llcSharers`
  for real shared-LLC contention; NUMA; private-L2-immune assumption (SMT/prefetch);
  worst-case vs measured.
- **RB4 Measurement/harness validity** — is the evidence sound? noise/signal
  discipline, the docker/cgeist FAILs, reproducibility, non-deterministic pass
  output, checksum-not-valid-IR. The methodology itself.

**Applicability**
- **AP1 Fusion/distribution + materialize tradeoff** — generalize
  keep-vs-recompute-vs-reload (DR + fission + the fusion/tiling forks) into one
  reuse-distance + effective-cache decision; the Halide compute_at/compute_root
  analogue.
- **AP2 Data layout & packing** — transpose, AoS↔SoA, padding/alignment, operand
  packing (a *codegen* red herring per memory — reconsider as a *cost-model* target:
  can the model COST a layout change via stride/cache-line/reuse-distance?), array
  contraction.
- **AP3 Tiling/temporal-blocking beyond GEMM** — stencils (temporal/time blocking),
  conv, general nests, multi-level blocking; extend working-set-vs-effective-cache
  tiling past BLAS-3.

## Method

Each track agent reads the current code + docs, does 2–3 targeted SOTA searches,
returns structured weaknesses/findings + ideas (each with change, impact,
tractability, spikeability, spike design, 3-column SOTA gap). Synthesis consolidates,
ranks by impact × tractability, builds the gap table, flags cross-track interactions
& contradictions, nominates the top spike-worthy items. Then I spike the top 1-2
inline (Ryzen core 8, robust measurement — repeats, checksum, out-of-noise).

## Findings

(Populated by synthesis → `COSTMODEL_ROBUSTNESS_FINDINGS.md`.)
