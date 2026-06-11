# Cost-Model Review Campaign

**Date:** 2026-06-08
**Author:** Claude (Opus 4.8), orchestrated for Tor
**Status:** ACTIVE — survey phase

## Goal

The most thorough cost-model review yet. **Fresh full-surface analysis** — prior
findings docs (`COSTMODEL_FINDINGS_CLAUDE.md` 2026-06-03, `COSTMODEL_SPIKE_FINDINGS.md`
2026-06-04) are **disregarded**: a lot of work has landed since (register-block family
selection, vector-dialect micro-kernel, tensor contractions, Polly comparison) and they
no longer describe the system. Analyze the cost model as it stands in the **current
code**.

Per-idea deliverable: **analytical weakness → expansion point / improved cost-calc method
→ mini gap-analysis vs SOTA**. The gap-analysis lets every idea be judged against the
state of the art rather than in isolation.

## Scope (user-selected tracks)

- **T1 — Vector-dialect vs SLP.** Model vector-dialect codegen cost *directly* instead of
  *assuming LLVM-SLP fires*. Replace the binary rank-gate (rank≥3→vector, ≤2→bet-on-SLP)
  with an analytical predictor of *when SLP fails*. The user's explicit steer: "a more
  direct mapping against the vector dialect is better than assuming SLP."
- **T2 — Cross-transform arbitration.** One analytical model that chooses *between* IR
  transformations (register-block / cache-tile / vectorize / fission) for a given loop
  nest. Today these live in disconnected passes with their own heuristics — no shared
  cost basis. The "between IR transformations" reading.
- **T3 — Cache/locality analytics.** Footprint, latency curve, reuse, associativity,
  working-set tier selection — the cache hierarchy cost core.
- **C4 — Register-pressure realism (CONSTRAINT, not a build target).** User's nuance:
  reg-pressure is hard to reason about before late-stage LLVM-IR. Use it *only* where it
  is an effective analytical guide. Map that valid subset — e.g. the vector-accumulator
  count of an *explicit* vector micro-kernel is a hard, analyzable invariant LLVM cannot
  undo (this is exactly what BLIS sizes its micro-kernel on); scalar spill prediction is
  not. C4 bounds what T1/T2 may legitimately assume about registers.

## SOTA baselines for the gap analysis

- **Analytical:** BLIS analytical model (Low et al. 2016, "Analytical Modeling Is Enough
  for High-Performance BLIS"), MLIR linalg vectorizer + LLVM `TargetTransformInfo` cost.
- **Polyhedral + roofline:** Pluto/Pluto+ cost, reuse-distance footprint, roofline &
  cache-oblivious analytical bounds.
- **Production heuristics:** LLVM TTI vectorizer cost, GCC/ICX cost heuristics — what real
  compilers actually ship.

## Method

1. **Survey (parallel, 1 deep-dive agent per track + the constraint).** Each agent reads
   the *current* code for its area, enumerates analytical weaknesses (location + formula +
   problem + severity), proposes expansion points / improved calc methods, and does 2-3
   targeted SOTA searches to ground a per-idea gap analysis. Honest, spike-first, no spin.
2. **Synthesize (1 agent, barrier).** Consolidate + dedup ideas, rank by impact ×
   tractability, build the consolidated gap-vs-SOTA table, flag cross-track interactions &
   contradictions, nominate the top spike-worthy ideas with a concrete spike design each
   (hypothesis, what to build, what to measure, expected signal, falsifier).
3. **Spike (serial, me).** Build & measure the top 1-2 ideas inline on Ryzen 7950X3D
   core 8, checksum-verified (`MATCH`), vs `clang -O3 -march=native -ffast-math` best loop
   order. No fan-out here — benchmarks contend on the box.

## Surface inventory (current code, 2026-06-08)

Cost-modeling lives in five places, two of which never talk to each other:

- **`CacheCostModel.cpp/.h`** (DataRecomputation core): `estimateComputeCost`
  (max(critical-path, ops/issueWidth=4)), `estimateLoadLatency` (4 hard size brackets:
  L1 4 / L2 12 / L3 40 / mem 200 cy), `estimateInterveningFootprint` (reuse-capped),
  `decideBufferStrategy`, `decideBufferElimination`.
- **`MemoryFission.cpp`** fission profitability: `keepCost = compute + 1 + nCons·L1lat`
  vs `recomputeCost = nCons·compute`; **assumes buffer fits L1** (loadLatency = l1Latency
  unconditionally).
- **`AffineRegisterBlock.cpp`**: family selection (`detectFamily` by operand stride),
  fixed 8×16 tile, dot-family 2×2/4×4, the **rank gate** (≥3→vector µkernel, ≤2→SLP),
  `worthTiling` cache-tile guard, VL-divisibility. **No connection to CacheCostModel /
  ArchHandler — pure local heuristics.**
- **`ArchHandler.h`**: unified combiner `α·mem + β·reg + γ·alu`, defaults (1,1,1).
- **`CpuCostModel.h/.cpp`**: per-op cycle costs, `minChainCost=15`, JSON override.

Key tension for this review: the **register-block / vectorization** decisions
(`AffineRegisterBlock`) — where the recent wins live — carry **no analytical cost model at
all**, while the **cache** cost model (`CacheCostModel`) is elaborate but drives a
different pass. T1/T2 are largely about giving the codegen decisions a real cost basis and
deciding whether the two worlds should merge.

## Findings

Complete → **`COSTMODEL_REVIEW_FINDINGS.md`**. Headline: `AffineRegisterBlock` (the
win-path) references the cost model ZERO times (grep-verified) — the cache model and the
codegen wins are disjoint universes. Highest-value work = give the win-path its first
analytic term, the BLIS-shaped explicit-vector micro-kernel cost (register-FIT + latency
floor), the one place register pressure is trustworthy. 5 ranked spikes; spike phase next.
