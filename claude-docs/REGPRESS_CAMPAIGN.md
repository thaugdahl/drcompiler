# Register-Pressure Campaign — Status & Reproduction

This document summarises the work done against
[`REGISTER_PRESSURE_PLAN.md`](REGISTER_PRESSURE_PLAN.md) and explains how
to reproduce the headline evidence the campaign has produced so far.

Scope of this document: **what shipped in code**, **what evidence exists
in the test suite**, and **what is still deferred to a real evaluation
on hardware**.  Honest verdicts only; do not consult this for hype.

---

## TL;DR

- All 5 phases of `REGISTER_PRESSURE_PLAN.md` landed as code + tests.
- The unified cost model produces decisions that **diverge from upstream
  MLIR's placeholders** under documented conditions — pinned by four
  lit tests.
- One of those flips happens under **realistic equal weights `(α=β=γ=1)`**
  with only the target FP register count tightened from 16 to 6 — this is
  the strongest target-aware-cost-model evidence the campaign has.
- **Runtime evidence on real kernels (PolyBench)** is the headline thing
  that's still missing.  Infrastructure to run it is in tree; pulling
  the trigger needs hardware time.

---

## What the campaign built

### Phase 1 — Arch handler infrastructure
`include/drcompiler/Analysis/{ArchHandler.h, RegisterClass.h, SpillStrategy.h}`,
`lib/Analysis/ArchHandlers/{Generic, X86_64_AVX2, X86_64_AVX512, ARM_Neon}.cpp`.
Four register classes (GP, FP, Vec, Pred — FP kept separate from Vec per
the §13.2 decision).  JSON dispatch via `ArchHandler::create(name)`.
Diagnostic pass `--print-arch-handler` prints the resolved handler +
budgets.

Tests: `test/Analysis/PrintArchHandler/` (8 lit tests).

### Phase 2 — Register pressure analysis
`include/drcompiler/Analysis/RegisterPressureAnalysis.h`,
`lib/Analysis/RegisterPressureAnalysis.cpp`,
`lib/Analysis/SpillStrategies/{ExcessHot, SumExcess, GraphColor}.cpp`.
Three full spill strategies (GraphColor is a real Chaitin-Briggs
implementation, ~150 LOC).  Works at any anchor op — the pass picks
FuncOp or ModuleOp scope by where it's invoked.

Tests: `test/Analysis/PrintRegisterPressure/` (20 lit tests covering
GP/FP/Vec/Pred classification across AVX2/AVX-512/NEON, all three
strategies, trip-count scaling, mask register handling).

### Phase 3 — Data-recomputation integration
Replaced `dr::decideBufferStrategy` with an arch-handler-combined
per-aspect (mem/reg/alu) breakdown.  Stripped `BufferElim`'s crude
`regBudget * spillCycles` arithmetic; routed it through
`RegisterPressureAnalysis::analyzeHypotheticalStatic` on the stored
value's parent function region.

Pass options refresh — 10 new (`dr-arch-handler`, `dr-spill-strategy`,
`dr-reg-budget-{gp,fp,vec,pred}`, `dr-spill-{reload,store}`,
`dr-cost-weight-{mem,reg,alu}`); 2 removed (`dr-reg-budget`,
`dr-spill-cycles`).

Tests: `bench/C/C037-regpress-under-budget.mlir`,
`bench/C/C038-regpress-over-budget.mlir` (updated to new options),
`bench/E/E015-elim-cost-regpressure.mlir`, `bench/E/E016-elim-cost-codebloat.mlir`
(updated keep/elim numbers under the new path), three new tests under
`test/Analysis/UnifiedCostModel/`.

### Phase 4 — Affine loop fusion fork
`lib/Transforms/AffineLoopFusionCostModel/LoopFusion.cpp` is a
verbatim fork of upstream MLIR's `LoopFusion.cpp` (SHA
`ce6d22760765e001a404d136c1d4dc1dce497791`, pinned in
`UPSTREAM.md`).  The placeholder cost model at line 657 of the
upstream file is replaced with `ArchHandler::combineCosts(mem, reg,
alu)` driven decisions.  Every divergence is marked `// DR-DIVERGE:`.

New pass `--dr-affine-loop-fusion` with 5 upstream options + 5
drcompiler-specific (`use-unified-cost-model`, `cpu-cost-model-file`,
`arch-handler`, `spill-strategy`, `emit-rationale`).

Tests: `test/Analysis/DrAffineLoopFusion/` (8 lit tests including the
decision-flip and rationale tests covered below).

### Phase 4.5 — Affine loop tiling fork
`lib/Transforms/AffineLoopTilingCostModel/LoopTiling.cpp` is a fork
of the same upstream tree.  The `nth_root(footprint / cacheSize)`
placeholder tile-size heuristic is replaced with a grid search over
`{2, 4, 8, 16, 32, 64}` scored by `ArchHandler::combineCosts`.

New pass `--dr-affine-loop-tile` with 4 upstream options + 4
drcompiler-specific (same names as the fusion fork's).

Tests: `test/Analysis/DrAffineLoopTile/` (5 lit tests including the
small-cache decision flip).

### Phase 5 — PolyBench evaluation scaffolding
- `scripts/polybench-bench.sh` extended with 7 new configs
  (`none`, `upstream-fuse`, `drcomp-fuse`, `drcomp-maxfuse`,
  `upstream-tile`, `drcomp-tile`, `drcomp-tile-fuse`).
- `scripts/calibrate_weights.py` — Nelder-Mead via scipy over
  `(α_mem, β_reg, γ_alu)` against a calibration kernel subset.
- `scripts/polybench_summary.py` — H1–H4 hypothesis evaluation
  (plan §8.2): drcomp avoids regressions, drcomp preserves wins,
  Pearson r against measured spills, weighted-sum vs hard-constraint
  ablation.
- `scripts/calibrate_regpress.py` — already in tree from Phase 2;
  now hooked into H3.

---

## Decision-flip evidence (the headline)

Four lit tests prove the unified cost model produces decisions that
diverge from upstream — without these, the whole campaign would be
speculation.

| Test | What it proves |
|---|---|
| `test/Analysis/DrAffineLoopFusion/decision-flip-tight-budget.mlir` | Under contrived `β_reg = 10⁴` weights the fusion fork rejects what upstream fuses — the unfused-baseline rejection path works end-to-end. |
| `test/Analysis/DrAffineLoopFusion/decision-flip-realistic-weights.mlir` | **Headline test.**  Under realistic equal weights `(α=β=γ=1)` with only `fp_budget=6` (vs AVX2's 16) the fork rejects fusion.  Same program, different target FP register count, different decision — exactly what a target-aware cost model is supposed to do. |
| `test/Analysis/DrAffineLoopTile/decision-flip-small-cache.mlir` | The tiling fork's grid search picks tile=2 outer + 128 inner where upstream's nth-root picks tile=8 uniformly.  Tile size selection is target-and-cache-aware. |
| `test/Analysis/DrAffineLoopFusion/emit-rationale-{fuse,reject}.mlir` | With `emit-rationale=true` the fork emits remarks with the actual `fused_total` / `unfused_total` numbers.  Cost-model regressions now fail at compile-test time rather than only at runtime. |

These tests are deterministic and bench-independent; they run in
`ninja check-drcompiler` like any other lit test.

---

## Reproducing the headline flip by hand

```bash
# Build (against Marco's bundled LLVM 22).
cmake -G Ninja -S . -B build-marco \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_INSTALL_DIR=/home/tor/Dev/marco/install/llvm-project
ninja -C build-marco dr-opt

# Upstream fuses the two loops -> one affine.for.
build-marco/tools/dr-opt/dr-opt \
  test/Analysis/DrAffineLoopFusion/decision-flip-realistic-weights.mlir \
  --affine-loop-fusion

# drcomp under fp_budget=6 rejects -> two affine.for survive.
build-marco/tools/dr-opt/dr-opt \
  test/Analysis/DrAffineLoopFusion/decision-flip-realistic-weights.mlir \
  --pass-pipeline="builtin.module(dr-affine-loop-fusion{
    cpu-cost-model-file=test/Analysis/DrAffineLoopFusion/Inputs/avx2-fp6-realistic.json
    emit-rationale=true})"
```

The second invocation prints the rationale on stderr:

```
fusion-rationale: REJECT fused_total=164 >= unfused_total=128
```

---

## Running the PolyBench evaluation (Phase 5, user-driven)

This is the missing piece.  Estimated wall time: ~hour, depending on
dataset and `--iters`.  Requires the `drcc` Docker image to be built.

```bash
# Full sweep (7 configs).
./scripts/polybench-bench.sh \
  --configs none,upstream-fuse,drcomp-fuse,drcomp-maxfuse,upstream-tile,drcomp-tile,drcomp-tile-fuse \
  --dataset LARGE --iters 7 \
  --csv results.csv

# Hypothesis tests on the CSV.
./scripts/polybench_summary.py --csv results.csv

# Optional: tune weights against a calibration subset.
./scripts/calibrate_weights.py \
  --polybench-dir third-party/polybench \
  --calibration "atax,bicg,gemm" \
  --holdout "2mm,3mm,syrk" \
  --csv calibration.log
```

The key questions Phase 5 answers:

- **H1:** On kernels where upstream's placeholder over-fuses, does
  drcomp recover?
- **H2:** On kernels where upstream's fusion is a legitimate win,
  does drcomp preserve it?
- **H3:** Pearson r between predicted spill cycles (from
  `RegisterPressureAnalysis`) and measured spills (from LLVM regalloc).
  Plan §2 set the gate at r > 0.85.
- **H4:** Does the weighted-sum combiner beat a hard-constraint
  ablation (where any over-budget pressure rejects fusion outright)?

---

## What is NOT validated

Be explicit about gaps so future-you doesn't claim more than the
evidence supports.

- **Runtime measurements on real kernels.**  All evidence is lit-test
  level so far.  Whether the cost model improves real PolyBench
  runtimes is open until §"Running the PolyBench evaluation" runs.
- **Phase 2 calibration canary.**  Per plan §11 the r > 0.85 gate is
  the canary for whether MLIR-level pressure tracks LLVM-level spills
  at all.  Eight high-pressure synthetic programs in `bench/regpress/`
  produced **zero LLVM spills at -O3** — LLVM's memory-operand
  scheduling and sret convention dissolve MLIR-level pressure.  The
  calibration is therefore not yet exercised; PolyBench's accumulator
  patterns are the next attempt.  If r stays low on PolyBench too,
  the approach is suspect and we pivot.  See
  [[regpress_decisions]] §"2026-05-29 — Phase 2 calibration deferred"
  for details.
- **Weight defaults.**  `(α, β, γ) = (1, 1, 1)` is the seed; tuning
  comes from `calibrate_weights.py` once real measurements exist.
- **GraphColor strategy in production paths.**  Implemented per §13.4
  and lit-tested for correctness; never been the active strategy in
  any DR-pass invocation that produced runtime evidence.

---

## Where to pick up

When resuming, the high-value branches are:

1. **Run Phase 5 PolyBench evaluation.**  Single most valuable next
   step.  See commands above.  The CSV + hypothesis report is the
   paper's empirical core.
2. **Mirror Phase 4d improvements to the tiling fork.**  Tiling
   currently lacks: unfused-baseline comparison, `emit-rationale`,
   slice-walked inner-region pressure.  These are direct ports of
   the fusion fork's recent improvements.  Code lives in
   `lib/Transforms/AffineLoopTilingCostModel/LoopTiling.cpp`.
3. **More decision-flip cases.**  Three flips in tree.  Could add
   GraphColor-vs-ExcessHot ablation, AVX-512 vs AVX2 contrasts,
   stencil patterns.  Each new flip is one more line of evidence.
4. **Honest re-read of `RegisterPressureAnalysis::queryHypothetical`.**
   I worried in passing that its conservative "add extra values to
   every program point" model overstates pressure in long loops.
   Worth a closer audit before relying on it for real numbers.

---

## Pre-existing bugs surfaced & fixed during the campaign

- `dr::estimateBufferSizeBytes` crashed on `memref<NxvectorMxfXX>`
  (hit `getIntOrFloatBitWidth()` on a vector type).  Fixed by
  branching on element kind in `CacheCostModel.cpp`.  Regression
  test: `test/Analysis/UnifiedCostModel/vector-memref-no-crash.mlir`.
- Fusion fork's `bestUnifiedTotal` was a `static` local that leaked
  across `isFusionProfitable` invocations.  Refactored to
  function-scope before adding the unfused-baseline comparison.
