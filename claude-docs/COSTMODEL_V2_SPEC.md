# Cost Model v2 + Transform Breadth — Implementation Spec

**Date:** 2026-06-10. **Author:** investigation session (Claude) for Tor.
**Audience:** a fresh implementation chat (Opus 4.8) with no prior context.
**Goal:** make the profitability/cost models of the drcompiler affine transform
suite less naive, and broaden the set of kernels the suite speeds up beyond
GEMM-shaped BLAS-3.

---

## 0. Ground rules (read first)

- **Repo:** `~/Dev/PhD/DRComp/drcompiler.git/onnx-mlir` (fork carrying upstream
  MLIR pass copies + DR extensions). Upstream LLVM pinned at
  `ce6d22760765e001a404d136c1d4dc1dce497791` (tree at
  `~/Dev/marco/source/llvm-project`). Divergences from upstream code are marked
  `DR-DIVERGE:` in comments — keep that convention.
- **Build:** `ninja dr-opt` in `onnx-mlir/build` (host build works; no
  container needed for compiling the pass library).
- **Tests:** llvm-lit tests under `test/` (`test/Analysis/DrAffineLoopTile/`,
  `test/Analysis/DrAffineLoopFusion/`, `test/AffineRegisterBlock/`, …). Add
  lit tests for every new cost-model behavior, especially `emit-rationale`
  output (existing pattern: `test/Analysis/DrAffineLoopTile/emit-rationale-tile.mlir`).
- **Commits:** local numbered checkpoints only — `costmodel_v2_1`,
  `costmodel_v2_2`, … **NEVER push.** Trailer:
  `Co-Authored-By: <model> <noreply@anthropic.com>`.
- **Branch base:** HEAD includes checkpoints `fusion_investigation_1..3`
  (816d84d, 0e08b4a, c49bd78): 4 fusion soundness fixes, the new
  `dr-affine-loop-distribute` pass, a pin-pass dominance fix, and a
  `rewrite-struct-memrefs.py` fix. Build on top of these.
- **Benchmark validity (critical):** all PolyBench benchmarking MUST use
  `cgeist -O0`. At `-O2` (and `-O1`) cgeist constant-folds `init_array`
  formulas into kernels and **deletes read-only input arrays** (gemm keeps 1
  of 3 arrays, 2mm 2/5, 3mm 3/7, bicg 2/5, gemver 3/9, trisolv 1/3). Every
  historical dense-LA number measured at -O2 (including the
  `results/pinned-2026-06-09` campaign and most `COSTMODEL_*_FINDINGS.md`
  docs in this repo) compares a synthetic compute-bound program against
  clang/Polly's real memory-bound one. Treat those numbers as *directional
  within the MLIR path only*. Stencil numbers (read-write arrays) are honest.
- **Correctness gate:** the `dr-pin-liveout` pass + `dr_observe_check.c`
  runtime print `SINK <hex-float>` at exit. Build a `none` reference at `-O1`
  (strict FP) and require **bit-identical** SINK from every transformed config
  before timing it. This caught two real miscompiles during the fusion
  investigation; never skip it.
- **Benchmark harness:** `~/Dev/PhD/DRComp/drcc-benchmarks/polybench/polybench-bench.sh`
  (configs in `CFG_PIPELINE`, lines ~125–169). Runs inside the
  `drcc-lean:x86_64` image; the image must be rebuilt when `dr-opt` or
  `tools/drcc/rewrite-struct-memrefs.py` change (the image bakes both).
  Single-kernel iteration is faster on the host: stage kernel → `cgeist` in
  container → host `dr-opt` → lower/link in container (pattern in
  `/tmp/fuseinv/batch.sh` from the prior session, may be gone; recreate as
  needed). Apply the harness's `sed 's/vector<\([0-9]*\)xi32>/vector<\1xi64>/g'`
  to cgeist output, run `rewrite-struct-memrefs.py`, and include
  `--convert-vector-to-llvm` in the mlir-opt lowering pipeline.
- **Background docs** in this repo worth skimming: `RATIONALE.md`,
  `COSTMODEL_SPIKE_FINDINGS.md` (mr×nr=8×16 validation),
  `TILING_INVESTIGATION_FINDINGS.md`, `POLYBENCH_FAMILY_FINDINGS.md`.

---

## 1. Current state — pass-by-pass inventory

### 1.1 `dr-affine-loop-tile` — lib/Transforms/AffineLoopTilingCostModel/LoopTiling.cpp (419 lines)

Upstream `affine-loop-tile` fork + a "unified cost model" tile-size search
(`getTileSizes`, lines 123–381).

**Band discovery:** `getTopLevelTileableBands` — only *maximal perfect nests
rooted at function top level*, legality via upstream `isTilingValid`. (The new
`dr-affine-loop-distribute` pass now turns PolyBench's imperfect nests into
such bands; before it, the tiler only ever strip-mined outer loops.)

**The model (`useUnifiedCostModel` path, lines 177–356):** grid search over
uniform candidates `{2,4,6,8,12,16,24,32,48,64}` scoring

```
perTileFP(c) = fp * c² / nExtent²          // fp = whole-band footprint bytes
lat(c)       = estimateLoadLatency(perTileFP(c), cache)   // 4/12/40/200 cyc tiers
memCycles(c) = (numIter / c) * lat(c)
liveVals(c)  = c²
regCycles(c) = liveVals > regCap ? numIter * (1 - regCap/liveVals) * spillCyc : 0
cost(c)      = alphaMem * memCycles + betaReg * regCycles
```

with `nExtent = numIter^(1/d)` and `regCap = vecBudget * (vectorWidthBits/64)`.

**Why it is naive — every assumption is GEMM:**
1. *Reuse factor `c` for every kernel.* `memCycles = numIter/c · lat` assumes
   each datum is reused `c` times per tile. True for matmul operands; false
   for streaming kernels (atax, bicg, mvt row-major streams, init loops),
   where actual reuse is 0 and tiling only adds min/max bound overhead.
   Because `numIter/c` decreases monotonically in `c`, the model essentially
   **always says TILE** once footprint > cache. Observed: identical rationale
   on every band, init loops tiled, `reg-block-tile` atax 0.55–0.60x.
2. *`liveVals = c²` accumulator block* — register-blocking term that only
   describes matmul-like accumulation.
3. *Cubical iteration space, 2D identity-accessed arrays, f64* — hardcoded.
4. *Uniform tile size across all band dims* — never the right answer for
   bands whose dims carry different reuse (e.g. tile k and j, not i).
5. *Per-band, not per-reference* — no access-pattern analysis at all; the
   only inputs are total footprint and trip counts.
6. The only REJECT paths: `excessFactor <= 1` (footprint already fits the
   configured cache) or untiled scoring best — which the formula's shape
   makes nearly impossible.

**Options:** `tile-size`, `tile-sizes`, `cache-size` (KiB),
`use-unified-cost-model` (default true), `cpu-cost-model-file`,
`dr-arch-handler`, `dr-spill-strategy`, `emit-rationale`, `separate`.

### 1.2 `dr-affine-loop-fusion` — lib/Transforms/AffineLoopFusionCostModel/LoopFusion.cpp (2128 lines)

Upstream greedy fusion fork (producer-consumer + sibling, fixed-point) with a
unified cost model in `isFusionProfitable` (lines 824–1148). Soundness was
fixed in `fusion_investigation_1`; the pass is now **sound but neutral** —
~1.00x on all 23+ kernels.

**Model:** per legal fusion depth, compute
`combine(memCycles, regCycles, aluCycles)` where `memCycles =
bytesToMemCycles(sliceWriteRegionBytes)` (cache-tier latency × lines touched),
`aluCycles` = fused op-instance count, `regCycles` =
`RegisterPressureAnalysis::analyzeHypotheticalStatic` spill estimate for the
cloned slice. Pick min-cost depth; reject if ≥ unfused baseline
(footprint+ALU+spill of both nests) or if redundant-compute fraction >
`compute-tolerance` (default 0.30). `combine` =
`ArchHandler::combineCosts(α·mem, β·reg, γ·alu)`.

**Gaps:**
1. **Read reuse is not modeled.** The memory term prices only the *slice
   write region*. Sibling fusion's entire profit — two nests reading the same
   large array, fusion halving traffic on it — contributes nothing to the
   score. This is the BLAS-2 opportunity (§3, WP4).
2. ALU term is op *instance counts* (`getComputeCost`), not CpuCostModel
   cycle weights — inconsistent with the rest of the suite.
3. No model of locality *improvement* in the fused body (the point of
   producer-consumer fusion): memory benefit check (lines 1103–1131) only
   compares footprint sums.
4. Stencil time-loop fusion (jacobi-style A→B, B→A) needs loop shifting —
   structurally out of reach of this pass; do not attempt here.

### 1.3 `dr-affine-loop-distribute` — lib/Transforms/DrAffineLoopDistribute.cpp (new, checkpoint 2)

Classical fission: splits a loop containing ≥2 top-level child loops into
clones, one child each, when no dependence at the loop's depth connects the
children. **Has no cost model — it always splits when legal.** This is the
pass that unlocked real tiling (gemm XL: distribute + tile = 0.78s vs Polly
1.23s), but unconditional fission is wrong in general: it destroys
producer-consumer locality between siblings that share data, and it is the
exact inverse of what fusion tries to do (run both blindly and they fight).

### 1.4 `affine-register-block` — lib/Transforms/AffineRegisterBlock.cpp (1378 lines)

GEMM-family pattern matcher: finds an innermost reduction loop with
k-invariant accumulators under two parallel spatial loops; unroll-jams i,j by
**fixed** `mr×nr` (8×16, validated within 5% of per-arch optimum in
`COSTMODEL_SPIKE_FINDINGS.md`); promotes accumulators to `iter_args`;
optionally emits explicit vector microkernels (broadcast family over j, dot
family over k, `vl` **hardcoded** 8/16). Handles imperfect nests by local
fission, triangular nests by peeling (head/diag, main/corner). Family select
shrinks the tile for dot kernels (syrk 4×4, syr2k 2×2) — the only
register-pressure-responsive decision, and it's a hardcoded table, not a
query of `RegisterPressureAnalysis`. Optional `cache-tile` (off by default)
uses its **own** `--l3-size/--llc-sharers/--mc/--nc/--kc` flags, not the
CpuCostModel JSON. Does not use CpuCostModel at all.

### 1.5 `data-recomputation` / `memory-fission` — lib/Transforms/{DataRecomputation,MemoryFission}.cpp

Recompute-vs-keep and materialize-vs-recompute decisions via
`CacheCostModel.h` (`decideBufferStrategy`, `estimateLoadLatency`,
`computeBufferElimCost`) + CpuCostModel op cycles + RegisterPressureAnalysis.
These have the most developed cost models in the suite and were ~neutral on
PolyBench (dr-cost 1.00x) — not the priority, but their shared infrastructure
(`CacheParams`, `estimateLoadLatency`, `kIssueWidth=4` critical-path/throughput
compute estimate) is what the tiler/fusion should reuse. Note duplicated cache
geometry: these passes take `--l1-size/--l2-size/...` pass options; tile/fusion
read the CpuCostModel JSON; reg-block has a third copy. Defaults agree today
(32KB/1MB/32MB, 4/12/40/200 cycles, line 64B) by convention only.

### 1.6 Shared infrastructure

- `drcompiler/Transforms/CpuCostModel.h` (+ .cpp, 237 lines): JSON-backed
  per-op cycle table (`ops`, `default_cost=5`), `cache{l1..mem latency/size}`,
  `arch{triplet, handler, vector_width_bits, spill_strategy,
  weights{alpha_mem, beta_reg, gamma_alu}}`, `registers{gp/fp/vec/pred
  budgets, spill cycles}`. The harness ships a probed `x86-64-avx512` JSON.
- `drcompiler/Analysis/ArchHandler.h` + `lib/Analysis/ArchHandlers/`:
  per-arch `combineCosts(mem, reg, alu)` + default params.
- `drcompiler/Analysis/RegisterPressureAnalysis.h`: static pressure traces
  over regions, hypothetical replay (`analyzeHypotheticalStatic`), spill
  strategies (excess-hot default).
- `drcompiler/Transforms/DataRecomputation/CacheCostModel.h`:
  `dr::CacheParams`, `dr::estimateLoadLatency(bytes, cache)`.
- Upstream affine analysis available to all passes: `MemRefRegion::compute`
  (parameterizable constraint-based footprints), `getMemoryFootprintBytes`,
  `checkMemrefAccessDependence`, `getLoopNestStats`/`getComputeCost`.

---

## 2. Benchmark evidence (what to beat / what to protect)

From `drcc-benchmarks/results/pinned-2026-06-09/SUMMARY.md` (cgeist -O2 —
see validity caveat; internal MLIR-path ratios still meaningful) and the
2026-06-10 single-kernel O0 experiments:

- Fusion configs: geomean 0.83x vs base at -O2 **before** the soundness fixes;
  ~1.00x everywhere after. Nothing left to lose, nothing yet gained.
- Tiling configs: 0.98–0.99x geomean (strip-mining noise). With distribute
  enabling real bands, the naive model's pointless-tiling cost shows up:
  atax `reg-block-tile` 0.55x at XL.
- reg-block: the one big winner (2mm 19x, bicg 4.4x, 3mm 2.8x at XL vs base)
  — but those -O2 numbers are inflated by the validity bug; re-baseline first.
- **O0 ground truth (gemm EXTRALARGE, 2026-06-10):** cgeist-base 3.11s,
  clang -O3 -march=native 2.79s, reg-block 1.75s, Polly 1.23s,
  **distribute + dr-affine-loop-tile 0.78s** (bit-identical checksum). This is
  the proof that band-enabling + tiling beats Polly when the model fires
  correctly; v2's job is to make that fire *only* where it helps.
- Polly at XL (geomean 1.57x vs clang) is the bar. Its biggest per-kernel
  margins over our best config were covariance, 3mm, bicg, atax, 2mm, trmm —
  i.e. **dense LA beyond gemm + shared-input BLAS-2**, exactly the breadth
  targets below.

---

## 3. Target breadth — PolyBench by class

| Class | Kernels | What should fire | Mechanism |
|---|---|---|---|
| BLAS-3 / matmul-like | gemm, 2mm, 3mm, syrk, syr2k, trmm, symm, doitgen | distribute → tile → reg-block | proven on gemm; needs per-kernel-correct tile decisions (triangular: syrk/trmm/symm — tiler currently needs constant bounds) |
| BLAS-2 shared-input | **atax, bicg, mvt, gemver, gesummv** | **sibling/input-reuse fusion; NO tiling** | atax: `y = Aᵀ(A·x)`; bicg: `s=Aᵀ·r, q=A·p`; mvt: two MV with A and Aᵀ. Two passes over the same large A; fusing the outer loops reads each A row once → up to ~2x traffic cut at memory-bound sizes. Polly gets these; we currently don't. Tiling must REJECT (streaming, zero temporal reuse). |
| Data mining | correlation, covariance | distribute → tile (+ reg-block on the matmul-like core) | covariance's core is a syrk-like triangular accumulation; Polly's largest win (6x vs our best at XL). |
| Stencils | jacobi-1d/2d, heat-3d, fdtd-2d, seidel-2d, adi | nothing new — protect neutrality | time-tiling needs skewing/diamond tiling: **out of scope**. Cost models must learn to say NO here (spatial tiling of one time step has no reuse to exploit at these sizes). |
| Solvers / factorizations | lu, cholesky, ludcmp, gramschmidt, durbin, trisolv | protect; opportunistic | triangular + in-place updates; fusion interleaving guard already skips them. Low priority. |
| Other | nussinov, floyd-warshall, deriche | protect | dependence-heavy; transforms should bail cleanly. |

“Protect” = config must stay within 0.97x of `none` per kernel and keep
bit-identical checksums.

---

## 4. Work packages

Ordered; WP0–WP2 are the core, WP3–WP4 the breadth, WP5–WP6 cleanup/stretch.

### WP0 — Re-baseline at cgeist -O0 (prerequisite, harness-side)

1. `polybench-bench.sh`: cgeist `-O2` → `-O0`; add `--convert-vector-to-llvm`
   to the mlir-opt lowering pipeline (reg-block on real -O0 kernels emits
   `vector.load`; the -O2 campaign never exercised it).
2. Add configs:
   `distribute-tile` = `func.func(dr-affine-loop-distribute,dr-affine-loop-tile)`,
   `distribute-regblock`, `distribute-tile-regblock`, and keep `none`,
   `drcomp-fuse`, `drcomp-tile`, `reg-block`, `clang`, `polly`.
3. Rebuild `drcc-lean:x86_64` (bakes dr-opt + rewrite-struct-memrefs.py; see
   memory note "probe-image-rebuild-chain" / `docker/` in drcc-benchmarks).
4. Validate all 30 kernels end-to-end at O0 (doitgen was expected-fail
   previously; recheck), then run the campaign once to establish the honest
   baseline table. **All WP acceptance numbers below refer to this baseline.**

### WP1 — Shared reuse analysis (`drcompiler/Analysis/ReuseAnalysis.h`)

The missing primitive behind every naive decision. For a perfect band (or any
loop list) and each memref *reference* in it, compute per band-loop ℓ:

- **invariant(ref, ℓ):** ℓ's IV is absent from the access map → full temporal
  reuse carried by ℓ (e.g. `A[i][k]` is invariant in j).
- **spatial(ref, ℓ):** ℓ steps the fastest-varying subscript with stride 1
  (cache-line reuse).
- **streaming(ref, ℓ):** otherwise (new line every iteration).

Plus: `reuseDistanceBytes(ref, ℓ)` = footprint of one iteration of ℓ over the
inner loops (via `MemRefRegion::compute` at depth of ℓ) — the data touched
between consecutive reuses; reuse is *realized* only if this fits the cache
level of interest. And `tileFootprint(band, tileSizes)` — exact per-tile
working set from `MemRefRegion` with tile-sized constant ranges substituted,
replacing the `fp·c²/n²` guess. API sketch:

```cpp
struct LoopReuse { bool invariant, spatialStride1; int64_t reuseDistanceBytes; };
struct BandReuseInfo {
  // per reference (load/store op) × band loop
  DenseMap<Operation*, SmallVector<LoopReuse>> refLoop;
  int64_t footprintBytes(ArrayRef<unsigned> tileSizes); // MemRefRegion-backed
  bool loopCarriesExploitableReuse(unsigned loopIdx, const dr::CacheParams &);
};
FailureOr<BandReuseInfo> analyzeBandReuse(ArrayRef<AffineForOp> band);
```

Implementation basis: each affine load/store's `AffineValueMap`; check IV
occurrence per result expr; stride of the last memref dim w.r.t. each IV.
Keep it constant-bounds-first (PolyBench is), return `failure()` otherwise so
clients fall back conservatively. Unit-test as a standalone test pass
(`-dr-test-reuse-analysis` printing per-ref classifications) with lit tests
for gemm, atax (no temporal reuse in the band loops' tiles), jacobi-2d.

**Acceptance:** lit tests demonstrating: gemm i/j/k each carry invariant reuse
for one of C/A/B; atax inner band reports no tiling-exploitable reuse;
init-style nests (`A[i][j] = f(i,j)`, no loads) report none.

### WP2 — Tiler v2 (rewrite `getTileSizes` + add a REJECT gate)

Replace the GEMM-shaped formula with reuse-driven decisions:

1. **Gate (new):** tile a band iff some loop ℓ that tiling would move carries
   `invariant` (or provably-aliasing group) reuse for ≥1 reference **and**
   `reuseDistanceBytes(ref, ℓ) > L_target` (reuse exists but is being evicted)
   **and** total band footprint > L_target. Otherwise `REJECT` with rationale
   `no-exploitable-reuse` / `already-resident`. This single gate kills init-loop
   tiling, atax/bicg/mvt tiling, and stencil spatial tiling.
2. **Per-dimension sizes:** loops carrying no reuse for any reference get tile
   size = trip count (i.e. untiled); the rest get sizes from a small grid
   search (reuse the existing candidate set per-dim, but score with
   `tileFootprint(band, sizes)` — the real `MemRefRegion` footprint — against
   the cache hierarchy, keeping the existing spill term for accumulator-like
   references only, i.e. references invariant in the innermost remaining
   loop). Cap the search combinatorics: per-dim candidates pruned to
   {trip, 16, 32, 64, 128} first, full grid only for bands of depth ≤ 3.
3. **Memory term:** `memCycles = Σ_ref linesTouched(ref, sizes) × lat(tier)`
   per tile × number of tiles, with `linesTouched` from the reuse classes
   (invariant → footprint once per tile; spatial → footprint/8; streaming →
   iterations). This replaces `numIter/c`.
4. Keep `tile-size`/`tile-sizes` explicit overrides bypassing everything
   (existing lit test relies on it), keep `emit-rationale` but make it
   per-band informative: which references/loops justified the decision.
5. Triangular/parametric bounds: gate returns REJECT today; leave a TODO for
   constant-bound-after-peeling integration with reg-block's peeling.

**Acceptance (single-kernel O0 at EXTRALARGE, vs `none`, checksums identical):**
gemm ≥ 3x (must not regress the proven 0.78s path); 2mm/3mm ≥ 2x with
distribute+tile(+reg-block); atax, bicg, mvt, jacobi-2d, all init-only nests:
REJECT fired, ≥ 0.97x; rationale remarks distinguishable per kernel (no more
one-verdict-fits-all). Lit tests for the gate on gemm vs atax vs init nest.

### WP3 — Distribute profitability (make fission enabling-driven)

Add a cost model to `dr-affine-loop-distribute` (currently always-split):

1. **Enabler mode (default):** split only when ≥1 resulting sub-nest becomes a
   *deeper perfect band* than before (depth increase ≥1) — i.e. fission that
   feeds the tiler/reg-block. Implement as a dry-run check: would
   `getPerfectlyNestedLoops` on each child grow after isolation?
2. **Locality guard:** when the children share a memref and the loop's
   combined per-iteration footprint fits L2, splitting doubles traffic on the
   shared array for no benefit — require either the enabler condition or
   working-set > effective LLC before splitting (reuse
   `dr::estimateLoadLatency`/`CacheParams`).
3. Pass options: `mode={always,enabler}` (`always` preserves today's behavior
   for A/B testing), `emit-rationale`.
4. Interaction note: fusion and distribution are inverses. Do **not** put both
   in one default pipeline blindly; benchmark configs keep them separate
   (fusion for BLAS-2 sibling reuse, distribute+tile for BLAS-3). A unifying
   driver is future work, not this spec.

**Acceptance:** gemm/2mm/3mm still split exactly as needed for WP2 numbers;
lu/jacobi (no enabling effect) untouched in `enabler` mode; lit tests for both
modes.

### WP4 — Fusion input-reuse term (the BLAS-2 breadth win)

Make `isFusionProfitable` see what sibling fusion is for:

1. Add a **shared-read traffic credit**: for sibling (and producer-consumer)
   candidates, compute the read regions of both nests
   (`MemRefRegion::compute` per load, unioned per memref); for memrefs read by
   *both* nests, fused traffic counts the overlap once while unfused counts it
   twice. Credit `bytesToMemCycles(overlapBytes)` against the fused total —
   but only when the overlap exceeds the cache tier that would hold it
   between the two nests (an L1-resident vector read twice is free either
   way; use `estimateLoadLatency(totalBetweenBytes, cache)` reasoning, i.e.
   credit only the part of the overlap whose intervening footprint evicts it).
2. Switch the ALU term to CpuCostModel-weighted cycles (`opCost` per op
   instance) for consistency (currently raw instance counts).
3. Verify the sibling-fusion path (`FusionMode::Greedy` includes a sibling
   stage) actually *finds* the atax/bicg/mvt/gemver pairs post-soundness-fixes
   — instrument with `emit-rationale` first; the prior neutral results may be
   "found but rejected by cost" or "never considered" (the interleaving guard
   from `fusion_investigation_1` must not be weakened — it killed a real
   miscompile; if it blocks a legal sibling case, refine it with a precise
   dependence check rather than removing it).
4. The four soundness guards (frame-shifted sink, `drIsSliceMaximal`,
   re-execution guard, interleaving guard) are load-bearing. Any cost-model
   change must keep the 13-kernel checksum sweep green.

**Acceptance (O0, XL, vs `none`):** ≥1.3x on at least two of
{atax, bicg, mvt, gemver}; no kernel < 0.97x; checksums identical on the full
suite; rationale shows the read-reuse credit firing on bicg and NOT on
unrelated-array siblings (e.g. 2mm's two matmuls share only the intermediate —
producer-consumer, not sibling-read).

### WP5 — Reg-block generalization (stretch)

Lower priority — it already wins. In order of value: (a) take `vl` from
`arch.vector_width_bits / elemBits` with the measured-AVX-512 override kept as
default (portability note at Passes.td:511–524); (b) replace the hardcoded
dot-family tile table (4×4/2×2) with a `RegisterPressureAnalysis` query on the
hypothetical unrolled body; (c) read cache geometry from the CpuCostModel JSON
instead of private `--l3-size/--llc-sharers` flags (folds into WP6). Do NOT
attempt new kernel families (MV/rank-1) until WP4 results are in — fusion may
already cover the BLAS-2 gap.

### WP6 — One source of truth for machine parameters

Every pass reads `CpuCostModel` (JSON or probed defaults) for cache geometry,
latencies, register budgets, and weights; per-pass `--l1-size`-style flags
become overrides that warn when they disagree with the JSON. Touches:
reg-block (own flags), MemoryFission/DataRecomputation (own flags),
LoopTiling/LoopFusion (already JSON). Mechanical; do it last so diffs stay
reviewable.

---

## 5. Validation protocol (applies to every WP)

1. **Correctness first:** for each touched config, full-suite SINK-checksum
   sweep vs `none` (-O1 strict-FP reference). Bit-identical or the change
   doesn't land. (Exception: configs that intentionally enable reassociation
   (`fast` flags in reg-block dot family) — compare against a reassoc-tolerant
   reference, as the harness already does.)
2. **Per-kernel timing matrix** at LARGE + EXTRALARGE, O0 harness, ≥3 reps,
   vs `none`/`clang`/`polly`. Geomean is the headline; per-kernel 0.97x floor
   is the regression gate.
3. **Rationale audit:** with `emit-rationale` on, dump decisions for all 30
   kernels and eyeball that they differ per kernel and cite the right reason
   (this is what exposed the v1 tiler — identical `tile_cost=625
   untiled_cost=10000` on every band).
4. **Lit tests** for every new decision path (accept + reject + override).
5. Checkpoint per WP: `costmodel_v2_<N>` with the WP's evidence in the commit
   message. Never push.

## 6. Explicit non-goals

- Stencil time-tiling (skewing/diamond/hexagonal) — separate project.
- A phase-ordering driver that arbitrates fusion vs distribution.
- Polyhedral scheduling (ISL-style); everything here stays within affine
  band analysis + existing MLIR utilities.
- Autotuning/search at runtime; the model stays analytic with JSON-probed
  machine parameters.
