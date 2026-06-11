# Cost-Model Findings (Claude)

**Date:** 2026-06-03
**Author:** Claude (Opus 4.8)
**Scope:** Investigation of the drcompiler cache-aware cost model — accuracy/applicability extension opportunities, real-world application classes that would benefit, and expected gains from applying affine transforms on MARCO.

---

## Implementation status (2026-06-03)

The four prioritized fixes from §3 were implemented. Up front, the honest framing
agreed before coding: **these do not produce real-workload speedups** — the
evidence (three NO-GO spikes, neutral PolyBench) says the profitable-transform
population on real code is thin. Their value is a *more truthful, validated, and
broadly-applicable* cost model. Each is grounded in code and locked with a test.

| Fix | What landed | Files | Verified by |
|---|---|---|---|
| **T1** — close calibration no-op | Inject `cpu-cost-model-file=` into the `dr-affine-loop-fusion`/`dr-affine-loop-tile` fork configs (they lack the `dr-cost-model=true` token the old injection keyed on). Writer/reader JSON keys confirmed aligned (`arch.weights.{alpha_mem,beta_reg,gamma_alu}` ↔ `CpuCostModel.cpp:189-191`). | `scripts/polybench-bench.sh` | dry-run of all 5 fork configs → valid pipelines; `dr-opt` parses+runs the injected forms end-to-end |
| **A1** — critical-path compute cost | `estimateComputeCost` now returns `max(criticalPath, ceil(totalOps/issueWidth))` via one memoized DAG walk, instead of the total op-sum. Linear dependent chains unchanged; wide independent expressions corrected. | `lib/Transforms/DataRecomputation/CacheCostModel.cpp` | new test `cost-model-critical-path.mlir` pins a 4-wide sqrt tree at `compute=22` (was 83) |
| **A2** — reuse-aware footprint cap | New `distinctMemrefBytes`/`cappedLoopFootprint`; each loop's `bodyFP × trip` is capped by the distinct static bytes its memrefs contain (temporal-reuse bound). Falls back to uncapped when a call or dynamic shape is present → never under-counts. Applied at all 5 loop-multiplier sites. | `lib/Transforms/DataRecomputation/CacheCostModel.cpp` | new test `cost-model-footprint-reuse-cap.mlir`: a 2 KB array re-read 256k× now scores `storeToLoadFP=4104` (was ~4 MB) → KEEP; dynamic-shape twin still RECOMPUTEs |
| **P1** — admit iter_args loops | Replaced the blanket iter_args reject in `loopBoundsAreEntryArgsOrConst` with an init-operand acceptability check (inits must be entry-arg/const → reproducible at caller); and disabled the unsound single-iteration extraction (F.1) whenever the region carries iter_args, forcing the sound full-loop region-clone (F.2). | `lib/.../Strategies/CrossFnOrdered.cpp` | new test `cross-fn-loop-iter-args.mlir`: an iter_args *scan* writer is now accepted and full-loop-materialized into a scratch buffer; output passes the verifier, no dangling cross-function refs |

**Validation.** Canonical suite (`build/test`) and `bench` rebuilt against the
same binary: **0 deterministic failures** introduced. Remaining non-green items
are pre-existing and unrelated to these changes: ~16 UNRESOLVED ONNX
model-intermediate tests and ~22 UNRESOLVED runtime/regpress tests (both require
external tooling/generated inputs), plus the flake below.

**Discovered (pre-existing, not caused by these fixes): non-deterministic pass
output.** `bench/C/C034-stride-no-loop.mlir` flakes — the line-27 load's
`partial-remat: REJECT_UNSAFE` remark is *sometimes emitted, sometimes not*,
depending on the order the pass visits loads (a pointer/ASLR-keyed `DenseSet`/
`DenseMap` iteration in `DataRecomputation.cpp`). The old (pre-change) binary
flakes identically, so this predates the fixes. It cannot be fixed at the test
level (the remark is sometimes absent, not merely reordered). **This is a real
truthfulness issue in its own right**: non-deterministic optimization decisions
mean benchmark/cost-model results are not reproducible run-to-run. Recommended
follow-up: make the driver's candidate-load iteration deterministic (sort by
source order / a stable key, or use `llvm::MapVector`). Left unchanged here to
stay within the agreed scope.

---

## 0. Verification note (what is grounded in code vs. delegated)

I read the following **directly** and the claims about them are first-hand:

- `include/drcompiler/Transforms/DataRecomputation/CacheCostModel.h` (full)
- `lib/Transforms/DataRecomputation/CacheCostModel.cpp` (full) — the load/keep/recompute decision core
- `include/drcompiler/Analysis/ArchHandler.h` (the unified-cost combiner + default weights)
- `lib/Transforms/AffineLoopFusionCostModel/LoopFusion.cpp` (cost-model wiring, lines 1750-1795)
- `scripts/polybench-bench.sh` (config table + cost-model-file injection logic)

The following are **delegated** to sub-investigations and cross-checked against the docs, but not every line was personally re-read: `Passes.td` numeric defaults, `gen_cpu_cost_model.py` / `calibrate_weights.py` internals, `MemoryFission.cpp` formula, the MARCO bench scripts, and the committed PolyBench result CSVs. Where a claim rests only on a doc or a delegated read, I say so. I have flagged the one load-bearing truthfulness claim (the calibration no-op, §2.1) as **personally code-verified**.

---

## 1. What the cost model is today (one-page map)

Two passes consume one cost model:

- **`data-recomputation`** — per-load decision: keep the buffered/loaded value, or recompute the stored expression. Core in `CacheCostModel.cpp`.
- **`memory-fission`** + the **`dr-affine-loop-fusion`/`dr-affine-loop-tile`** forks — materialize-vs-recompute across sibling loops, plus a unified cache+register+ALU combiner (`ArchHandler::combineCosts`).

### The decision functions (verified in `CacheCostModel.cpp`)

```
decideBufferStrategy:
  effLoadLat   = estimateLoadLatency(bufferSize + storeToLoadFootprint)
  memKeep      = numConsumers * effLoadLat
  aluKeep      = aluCost + 1                       // "+1" = single store, in ALU units
  memRecompute = numConsumers * (leafLoadCost + operandPenalty)
  aluRecompute = numConsumers * aluCost
  total{Keep,Recompute} = arch.combineCosts(mem, reg, alu, params)   // α·mem+β·reg+γ·alu
  recompute  ⟺  totalRecompute <= totalKeep        // ties → recompute

decideBufferElimination:
  keep = numLoads*loadLat + numStores*storeLat + allocOverhead + capacityPenalty
  elim = numDistinctComputes*perElemCompute + codeBloat + regPressure
  eliminate  ⟺  elim <= keep
```

### What it currently models

| Dimension | How | Where |
|---|---|---|
| Cache hierarchy | 4 hard size brackets → fixed latency (L1 4 / L2 12 / L3 40 / mem 200 cy) | `estimateLoadLatency` |
| Working-set eviction | bytes of intervening memory traffic pushes the buffer into a higher latency tier | `estimateInterveningFootprint`, `decideBufferStrategy` |
| ALU/compute cost | weighted **sum** of ops in the SSA operand tree (add 1, mul 3, div/rem 15, transcendental 20; loads free) | `estimateComputeCost` |
| Consumer fan-out | linear in `numConsumers` on both sides | `decideBufferStrategy` |
| Stride / cache line | partial-remat leaf cost `effBytes = min(line, stride·elemSize)` | (DR pass leaf path) |
| Register pressure + spill | `regCyclesKeep/Recompute` from a separate `RegisterPressureAnalysis`, combined per-arch | `ArchHandler`, `decideBufferStrategy` |
| Buffer-elimination rollup | adds store traffic, alloc overhead, code-bloat (i-cache), reg-pressure, CSE discount via structural hash | `decideBufferElimination` |
| Per-arch tuning | `α/β/γ` combiner weights, reg budgets, vector width; JSON-overridable | `ArchHandler.h`, `CpuCostModel` |

Defaults (delegated, from `Passes.td` / `ArchHandler.h`): `kDefaultTripCount=128`; combiner weights `α=β=γ=1.0`; reg budgets gp/fp/vec = 16/16/16, spill reload/store = 5/1 cy.

---

## 2. Truthfulness ledger (read this first)

The docs already catalogue many *accuracy* gaps honestly (`docs/recomputation-analysis.md` §3, `research/PGO.md`). The findings below are the ones that bear on whether the cost model's reported results mean what they appear to mean. They are ordered by how much they undermine a claim of "the cost model decided X."

### 2.1 The weight calibration is a no-op in the fork path — **code-verified**

This is the single most important truthfulness issue, and it is mechanical, not a matter of opinion:

- `calibrate_weights.py` runs Nelder-Mead over `(αmem, βreg, γalu)`, evaluating each candidate by running `polybench-bench.sh` on the `drcomp-tile-fuse` config and reading its median runtime.
- `polybench-bench.sh:256` attaches the calibrated JSON **only when the pipeline string contains the literal token `dr-cost-model=true`**.
- The fork configs it calibrates — `drcomp-fuse`, `drcomp-tile`, `drcomp-tile-fuse` (`polybench-bench.sh:141-145`) — contain `dr-affine-loop-fusion`/`dr-affine-loop-tile`, **not** `dr-cost-model=true`. So `COST_MODEL_FILE` is never injected.
- `LoopFusion.cpp:1764-1767`: with `cpuCostModelFile` empty, the pass falls back to `CpuCostModel::getDefault()` and `archHandler->defaultParams()`, i.e. weights `(1,1,1)` (`ArchHandler.h:37-39`, combiner at `:86-94`).

**Consequence:** every Nelder-Mead candidate runs the *identical* default-weight binary. The objective is constant in the parameters it is optimizing. No calibrated weight set has ever reached the benchmarked binary in the fork path. Any plot or table captioned "calibrated cost model" in the fusion/tiling experiments is, in fact, the `(1,1,1)` default model. The unified combiner is therefore **unvalidated** — its α/β/γ degrees of freedom have never been exercised end-to-end.

This also explains the "bench-vs-manual" discrepancy recorded in the project memory: there is (a) this silent no-op, and (b) a separate, environmental quoting failure where `'dr-affine-loop-fusion' does not refer to a registered pass` appears **only** through the harness's `docker run … bash -c "…"` wrapper while the byte-identical pipeline parses fine when run directly. (b) is *not* a pass-registration bug — `dr-opt` registers all forks via one `registerDRCompPassesPasses()` call; it is shell quoting/expansion in the wrapper. Fixing (a) is a precondition for the calibration story to mean anything; fixing (b) is a precondition for the sweep to run at all.

### 2.2 The micro-benchmark wins are designed-in, not predicted

The canonical "wins" (`benchmarks/div_chain.c`, `sqrt_consumers.c`, `transcendental_chain.c`; `bench/runtime/R*`) all fix `N=4000` *specifically so the fission buffer lands in L1* and place the REPS loop inside the kernel so fission hits the hot path (per the file headers / delegated read). With the buffer guaranteed L1-resident, the model's flat-tier latency happens to be right by construction. `GAP.md`'s own table shows the model **under-predicts** the measured speedup (4 consumers: 2.69× predicted vs 2.88× measured) and was only validated in that L1-resident regime. So the model is demonstrated to be *directionally* correct on hand-sized kernels, not *quantitatively* accurate, and not validated outside L1 residency. That is a much weaker claim than "the cache cost model predicts speedups."

### 2.3 The only real-application numbers are non-regression, plus one outlier

The sole committed measurements are PolyBench CSVs (delegated read of `benchmarks/results/`): DR configs track clang/cgeist within noise on dense LA, with **one** kernel — floyd-warshall — ~9% faster under `dr-cost`/`dr-partial`. No committed numbers for MARCO, ONNX, or SPEC. This is consistent with the project memory (all three speedup spikes NO-GO). The honest summary of the evidence base is: *"the passes do not regress dense linear algebra, and occasionally help one irregular kernel."* That is a real, defensible result — it is just not a speedup story.

### 2.4 The cost model is coarse where it most needs to be sharp — three structural inaccuracies (verified)

1. **Compute cost ignores ILP.** `estimateComputeCost` does `cost += opCost(defOp)` over the operand DAG — a *total op count*. A chain of 10 dependent adds and 10 independent adds both cost 10, but on an out-of-order core the latter is ~2-3 cy. Because the recompute side scales as `numConsumers · aluCost`, over-counting `aluCost` systematically biases the model **toward buffering** (and toward fission). `PGO.md` admits the same via llvm-mca. This is the accuracy fix with the broadest blast radius: it touches every recompute/keep/fission decision.

2. **Footprint double-counts reuse; no spatial or temporal locality.** `estimateOpFootprintBytes` charges `elemBytes` *per access*, and `estimateBlockFootprintBytes` multiplies body footprint by trip count with **no deduplication of repeated cache lines**. A loop touching the same 64-byte line 1000× is scored as 64 KB. A unit-stride stream and a random gather over the same array score identically. Since this footprint feeds the latency-tier selection in `decideBufferStrategy`, the over-count can wrongly promote a buffer to a slower tier and flip the decision. This is the second-broadest accuracy fix.

3. **Flat latency step function.** `estimateLoadLatency` is four hard brackets: a 31 KB buffer costs 4 cy, a 33 KB buffer 12 cy — a 3× cliff across a 2 KB boundary, with no associativity, set-conflict, or miss-ratio curve. Cheap to smooth; matters most near the L1/L2 boundary where many real buffers sit.

### 2.5 Secondary truthfulness gaps (verified or delegated)

- **Keep-side write-back is unpriced in `decideBufferStrategy`.** `aluKeep = aluCost + 1` treats the store as one ALU op; the *memory* write-back of a buffer that exceeds cache is never charged on the keep side. `decideBufferElimination` does charge `numStores·storeLat`, so the two paths disagree on whether stores cost memory.
- **Cross-function footprint is pinned flat to `l2Size`.** `estimateInterveningFootprint` returns `cache.l2Size` for any store/load pair in different functions, and `estimateOpFootprintBytes` returns `l2Size` for any dynamic-shaped call operand — a single magic constant standing in for interprocedural reality.
- **Trip-count fallback `128` is arbitrary** and applied identically to a 4-iteration and a 4-million-iteration loop when bounds are non-constant.
- **Calibration probe never reaches production.** `gen_cpu_cost_model.py --probe` output is not wired into `drcc`'s default `DR_PASS_FLAGS`; SPEC runs use the hardcoded ~Haswell defaults. And the JSON schema has no `cache_line` key (delegated), so even a probed model cannot change line size from 64.
- **The register term is unfalsified.** Per the campaign notes, the reg-pressure validation canary produced zero LLVM spills on 8 high-pressure synthetic kernels at `-O3`, so the `βreg` term the (no-op) calibrator is meant to tune has never been shown to track real spills.

---

## 3. Cost-model extension opportunities

Prioritized by (truthfulness/accuracy impact) × (tractability), with the concrete change site. I separate **accuracy** (make predictions match reality), **truthfulness/calibration** (make the reported model the model that ran, and validate it), and **applicability** (cover more code patterns).

### Priority table

| ID | Class | Change | Impact | Effort | Site |
|---|---|---|---|---|---|
| T1 | Truthfulness | Make fork configs actually receive the cost-model JSON (close the §2.1 no-op) | **Critical** | Low | `polybench-bench.sh:256`, fork configs `:141-145` |
| A1 | Accuracy | Critical-path compute cost (ILP), not total op sum | **High** | Med | `estimateComputeCost` `CacheCostModel.cpp:21-49` |
| A2 | Accuracy | Cache-line-granular, reuse-deduped footprint | **High** | Med | `estimateOpFootprintBytes` / `estimateBlockFootprintBytes` |
| P1 | Applicability | Lift `iter_args` rejection in `buildLoopPlan` (unblocks reductions/time-stepping → MARCO, RNN, stencil) | **High** | Med-High | DR pass `buildLoopPlan` (ARCHITECTURE.md:288-292) |
| T2 | Truthfulness | Wire `--probe` output into `drcc` default pipeline; validate, don't assert | High | Low | `drcc.sh.in` / `DR_PASS_FLAGS`; `calibrate_regpress.py` canary |
| A3 | Accuracy | Continuous / associativity-aware latency curve | Med | Low | `estimateLoadLatency` `:100-107` |
| A4 | Accuracy | Charge write-back on keep side when buffer > L1 | Med | Low | `decideBufferStrategy` |
| P2 | Applicability | Affine-store index coverage via `AffineValueMap` → more SINGLE | Med | Med | provenance (affine.store → nullopt today) |
| A5 | Accuracy | Converge PartialRemat dual gate + S4 amortized gate on `decideBufferStrategy` | Med | Med | ARCHITECTURE.md:276-305 |
| T3 | Truthfulness | Add `cache_line` to JSON schema; reconcile the two ALU tables | Low | Low | `CpuCostModel` JSON parser |
| P3 | Applicability | Vector-aware op cost (cost/lane by `vectorWidthBits`) | Med | Med | `CpuCostModel::opCost` + `estimateComputeCost` |
| P4 | Applicability | Transitive interprocedural footprint (callee mod-ref summary) | Med | High | `estimateInterveningFootprint` |

### The three highest-leverage, in detail

**T1 — close the calibration no-op (do this first; it is the cheapest credibility win).**
Either (a) have `polybench-bench.sh` inject `cpu-cost-model-file=…` for `dr-affine-loop-*` configs unconditionally (not gated on the `dr-cost-model=true` token, which those configs never carry), or (b) add a `cpu-cost-model-file` option pass-through and key the injection on the config name. Until this lands, every "calibrated" fusion/tiling result is the `(1,1,1)` default and should be relabeled as such. After it lands, re-run `calibrate_weights.py` and confirm the objective actually varies with `(α,β,γ)` (a quick sanity check: two distinct weight vectors must produce two distinct runtimes; today they cannot).

**A1 — critical-path compute cost.**
Replace the accumulate-everything loop in `estimateComputeCost` with a memoized longest-dependency-path over the SSA DAG (depth where each op contributes its latency, loads/block-args = 0). Keep total-op-count as a *throughput* term and take `effectiveCost ≈ max(criticalPath, totalOps / issueWidth)`, with `issueWidth` from `ArchParams`. This is the documented #1 source of systematic bias (over-pricing recompute → over-buffering), it is locally contained to one function, and it is directly validatable against `llvm-mca` latency for the same op tree (the calibration harness already shells out to llvm-mca). Expected effect: more loads classified recompute-favourable, fewer spurious fissions on cheap dependent chains.

**A2 — reuse-aware, cache-line-granular footprint.**
For an affine access with constant stride `s` over `t` iterations of span `span` (cap at the static array size `A`), charge *distinct lines*: `bytes = min( ceil(min(t·s, span)·elemSize / line) · line, A )` instead of `t · elemSize`. This fixes spatial locality (stride) and temporal reuse (cap at array size) in one stroke. The affine machinery to recover stride/span (`AffineValueMap`) is already referenced in the roadmap and partly used by the affine-footprint analysis (recent `Add affine footprint analysis` commit). Because footprint feeds tier selection, this is the change most likely to *flip* real decisions toward the correct side, and it is the prerequisite for the model to mean anything on streaming/stencil code (§4).

**P1 — lift the `iter_args` rejection.**
`buildLoopPlan` bails on loops with loop-carried values. That excludes every reduction, every Euler/RK time-step, every RNN recurrence, and most stencils — i.e. essentially all of MARCO's hot loops and much of ONNX. ARCHITECTURE.md itself calls this "the next big win for Marco workloads where reduction loops dominate." This is the single change that converts the cost model from "applies to toy store→load chains" to "applies to the motivating workload." It is the highest-impact *applicability* item, gating §4 and §5 below.

### A note on what is *not* worth doing

- **Per-op latency micro-tuning** (the SKYLAKE preset, transcendental sqrt-multiples) is polishing constants while the structural errors in §2.4 dominate. Low ROI until A1/A2 land.
- **More PolyBench tuning.** `research/NO_POLYBENCH.md` is correct: dense LA has read-only hot inputs (no store→load chain) and reduction provenance that breaks at outer-loop joins; "even with perfect analysis the cost model would reject every candidate." Keep PolyBench only as the non-regression baseline it already serves as.

---

## 4. Real-world application classes (MLIR-compilable) that the cost model can help

The genuine sweet spot is narrow and specific: **store→load chains where the stored expression is cheap relative to a cache miss**, and **expensive pointwise work duplicated across sibling loops** (the fission/Halide-`compute_inline` case). Reductions over read-only inputs (dense LA) are out, by construction. Within that frame:

| Class | MLIR path | Why it fits (or not) | Status / what's needed |
|---|---|---|---|
| **Modelica / MARCO physics simulation** | `marco -fdata-recomputation` (pass compiled into a MARCO fork) | Time-stepped ODE/DAE systems re-evaluate the same transcendental physics sub-expressions (exp/sqrt/pow) across many equations each step — the duplicate-pointwise-across-consumers pattern. | **Blocked today** by P1 (iter_args). The motivating workload, but the pass bails on its hottest loops. See §5. |
| **ONNX / ML inference — elementwise & normalization layers** | onnx-mlir Krnl → Affine → `dr-opt` | Activation functions (gelu/sigmoid/softmax-exp), layernorm/softmax recompute-vs-load, LUT-vs-recompute. These are pointwise chains shared across consumers. | Working path; needs A1 (transcendental chains are dependency-structured) and ideally P3 (these layers get vectorized). The GEMM/conv core is **NO-GO** (read-only inputs) — target the *epilogue/normalization*, not the matmul. |
| **Image / signal pipelines (Halide-style)** | C → cgeist → Affine | Pointwise stages (gamma, color transform, transcendental filters) consumed by multiple downstream stages — textbook materialize-vs-recompute. The fission model *is* a `compute_inline`/`compute_root` decision. | Strong structural fit. Needs A2 (these stream with unit stride; current footprint over-counts massively) and P1 (multi-stage pipelines carry state). |
| **Stencils / finite-difference PDE (weather, seismic, CFD)** | C/Fortran→cgeist, or MARCO | Halo recompute-vs-store; repeated coefficient evaluation. | Needs P1 (loop-carried) + A2 (spatial locality) to be modeled correctly at all. High potential, currently mismodeled. |
| **Transcendental-heavy DSP / scientific kernels** (FFT twiddles, special functions, Monte-Carlo) | C→cgeist | Recompute-vs-table (`R27` LUT case already encodes this); expensive pointwise reused. | Working for the leaf/LUT case; A1 makes the break-even honest. |
| **SPEC CPU 2017 subset** — `lbm`, `nab`, `imagick`, `x264`, `mcf` | `drcc` drop-in `cc` | Irregular, interprocedural C with reused pointwise/transcendental work (named in `NO_POLYBENCH.md` as the right DR targets). | The intended evaluation target; gated by cgeist survival, not by the cost model. |
| **Dense linear algebra / PolyBench** | — | Read-only hot inputs, reduction provenance. | **NO-GO** as an optimization target. Keep as non-regression baseline only. |

**The cross-cutting observation:** every class in the "fits" rows shares two requirements — (1) the analysis must handle loop-carried state (**P1**), and (2) the footprint must respect spatial/temporal locality (**A2**) or it will over-charge streaming access and reject good candidates. Those two extensions are the gate to *all* real-application applicability, not incremental nice-to-haves. The current micro-benchmarks avoid both by construction (no iter_args, L1-resident), which is exactly why they pass and real code does not.

---

## 5. Expected gains from affine transforms on MARCO

**Integration (delegated, cross-checked):** unlike the ONNX textual-`.mlir` path, the DR/fission machinery is compiled *into* a MARCO fork and exposed as `marco -fdata-recomputation [-dr-cost-model]`; the bench harness (`marco-dr-bench.sh`) times Modelica `.mo` models (default corpus = MARCO's Euler-forward simulation tests, e.g. `CycleWithDerivative`), scaling `end-time ×100` to get timeable runs. This is a cleaner integration than ONNX (no DLTI sed fixup, runs inside MARCO's own MLIR pipeline).

**The affine transforms in scope** (the project already has `dr-affine-loop-fusion` and `dr-affine-loop-tile` forks plus memory-fission): cost-model-gated **fusion** (improve temporal locality of state vectors across equation loops), **fission** (materialize a shared expensive physics term once instead of recomputing per consumer), **recomputation** (forward stored intermediates within a step), and **tiling** (over the spatial/equation dimension within a step — *not* across time, which is sequential).

### Honest expectation

I will not predict a speedup. The evidence says to be conservative, and there is a hard blocker:

1. **Expected gain today ≈ 0.** Modelica time-stepping is inherently loop-carried (state advances each Euler/RK step; residual/reduction loops dominate). `buildLoopPlan` rejects `iter_args`, so the pass **bails on MARCO's hottest loops**. Until **P1** lands, the cost model never even runs on the code that matters. Any current MARCO measurement is measuring the pass *declining to fire*.

2. **After P1, gains are workload-bimodal, and most of the mass is in the neutral/NO-GO half.**
   - *Transcendental-heavy physics models* (heat transfer, fluid, reaction kinetics — repeated exp/sqrt/pow shared across equations) match the fission sweet spot. The synthetic analog (`transcendental_chain.c`) shows large micro-kernel speedups, but those are L1-resident by design; on real models with state vectors that exceed L1, expect *single-digit-percent* effects at best, and only after **A2** stops over-charging the streaming state access.
   - *Linear / sparse-system-dominated models* (the common case) are reduction-like — solver-bound, read-mostly. These are PolyBench-shaped and therefore **NO-GO** for the same reasons; the cost model should (correctly) decline.
   - Net: a *subset* of models could win modestly; many will be neutral; the aggregate "MARCO speedup" number is likely small and noisy. That is the truthful prior given the three NO-GO spikes.

3. **The defensible MARCO contribution is a reframe, not a speedup.** The RATIONALE thesis is that cgeist/MARCO fuse greedily with *no cost model*. The strongest honest claim is **cost-model-gated fusion that avoids over-fusion regressions** — "don't regress, occasionally win," which is exactly what the PolyBench non-regression data already supports. Positioned as *safe-fusion gating for Modelica* (a place where a wrong fusion costs real cache traffic and there is currently no model at all), this is a credible result that does not depend on finding a speedup that the spikes suggest is not there.

### Minimum path to a *measurable* MARCO result

1. **P1** — lift `iter_args` (otherwise nothing fires).
2. **A2** — reuse-aware footprint (otherwise state-vector streaming is over-charged and everything is rejected).
3. **A1** — critical-path compute (otherwise transcendental chains are over-priced, biasing toward buffering).
4. **T1** — so the run actually uses the calibrated/probed model and the number is attributable.
5. Then measure on the transcendental-heavy subset of the MARCO test corpus, reporting **predicted-vs-measured** per model (the validation `GAP.md` started), and report the neutral/NO-GO models honestly rather than filtering to the winners.

Steps 1-4 are the same extensions that gate every other real application class (§4) — MARCO is not a special case, it is the **first real workload that exercises the whole list at once**. That is its value: it forces the cost model to become true.

---

## 6. Recommended next steps (in order)

1. **T1** — close the calibration no-op and relabel past fork-config results as default-weight. *(Low effort, restores credibility of every fusion/tiling claim.)*
2. **A1 + A2** — critical-path compute cost and reuse-aware footprint. *(The two structural accuracy fixes that gate all real-application applicability; both validatable against llvm-mca and against the affine footprint analysis already in `lib/Analysis/`.)*
3. **P1** — lift `iter_args`, unblocking MARCO/stencil/RNN/reduction. *(Highest applicability leverage.)*
4. **T2** — wire `--probe` into the production pipeline and get the reg-pressure canary to actually spill, so `βreg` is falsifiable.
5. Re-position the MARCO and PolyBench stories as **cost-model-gated safe fusion (non-regression + occasional win)**, with per-kernel predicted-vs-measured tables, rather than an aggregate speedup pitch.

The throughline: the cost model's machinery (unified combiner, arch handlers, calibration) is more elaborate than its *validated* core. The cheapest, most honest progress is to (a) make the model that runs be the model that is reported (T1/T2), (b) fix the two structural inaccuracies that bias every decision (A1/A2), and (c) unblock the one applicability gate (P1) that lets any real workload reach the model at all.

---

## Appendix: key file:line index

- Decision core: `lib/Transforms/DataRecomputation/CacheCostModel.cpp` — `estimateComputeCost` (~:21-49, op-sum, no ILP), `estimateLoadLatency` (~:100-107, 4-bracket step), `estimateInterveningFootprint` (cross-fn → `l2Size`), `estimateOpFootprintBytes` (elemBytes/access; dynamic call operand → `l2Size`), `estimateBlockFootprintBytes` (`bodyFP × tripCount`, no dedup), `decideBufferStrategy`, `decideBufferElimination`.
- API + constants: `include/drcompiler/Transforms/DataRecomputation/CacheCostModel.h` (`kDefaultTripCount=128`, `MaterializationInputs`, `BufferElimCostInputs`).
- Unified combiner: `include/drcompiler/Analysis/ArchHandler.h:32-94` (`ArchParams` weights default `1.0`; `combineCosts` = `α·mem+β·reg+γ·alu`).
- Calibration no-op: `scripts/polybench-bench.sh:141-145` (fork configs), `:256` (injection gate), `lib/Transforms/AffineLoopFusionCostModel/LoopFusion.cpp:1764-1767` (empty file → `getDefault()`).
- Known gaps already documented: `docs/recomputation-analysis.md` §3, `research/PGO.md`, `ARCHITECTURE.md:276-305` (PartialRemat dual gate, S4, `buildLoopPlan` rejects `iter_args` `:288-292`).
- App-class verdicts: `research/NO_POLYBENCH.md` (dense LA NO-GO, SPEC the target, `lbm/mcf/nab/imagick/x264`).
- MARCO: `scripts/marco-dr-smoke.sh:48-66`, `scripts/marco-dr-bench.sh:38-40`.
- Evidence base: `benchmarks/results/*/results.csv` (PolyBench non-regression + floyd-warshall ~9%); `GAP.md:120-131` (fission predicted-vs-measured, under-prediction).
