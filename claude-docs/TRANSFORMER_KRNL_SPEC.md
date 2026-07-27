# TRANSFORMER_KRNL_SPEC — Transformer GEMM Shape Coverage

**Status:** Draft spec (2026-06-16). Next codegen-campaign direction.
**Owner decisions locked** (see §2). **Default OFF / byte-identical** until a WP lands behind its gate.

> Naming note: the file says `KRNL` because the *origin* and the *competition* are
> onnx-mlir's `krnl.matmul` optimization path (the thing we lose to on transformers).
> The **injection point is the affine level** (decision §2.3) — we extend
> `affine-register-block` + the demote/promote pipeline, we do **not** hook the krnl
> dialect. The name marks the problem domain, not the implementation altitude.

---

## 1. Motivation — the measured gap

End-to-end inference, host machine-model codegen (`dr-scalar-reduction-demote →
affine-register-block → dr-scalar-reduction-promote`) vs onnx-mlir `--O3`, identical
inputs, correctness-checked (top-1 identical, norm-rel-err < 1e-4). Harness:
`scripts/onnx-codegen-bench.sh` (extended this session for 2-input transformers via
`--inputs "i64:1,128;f1:1,128"`).

| model | kind | none (naive) | **codegen (ours)** | o3 | verdict |
|-------|------|-------------|--------------------|-----|---------|
| resnet50 | CNN/conv | 2.103 s | **0.934 s (2.25×)** | 1.294 s | **codegen beats o3 by 1.39×** ✅ |
| gpt-neox | transformer | 5.93 ms | 3.95 ms (1.50×) | **2.68 ms** | o3 1.47× faster ❌ |
| openai-gpt | transformer | 20.25 s | 20.76 s (**0.98×**) | **0.555 s** | o3 **36.5×** faster — codegen ≈ no-op ❌ |
| mnist | tiny CNN | 19 µs | (µs-noise) | (µs-noise) | functional only |

The headline: **on the conv-heavy CNN our register-block wins; on transformers the
pass is ~a no-op** (openai-gpt 0.98× — it literally does nothing), leaving the naive
20 s lowering exposed while onnx-mlir's krnl matmul path runs in 0.55 s. This spec
closes that.

> **Honesty flag carried from the mapping phase:** the *36× vs 1.47×* split is
> **inferred, not measured** — no doc explains why openai-gpt loses 36× but gpt-neox
> only 1.47×. The leading hypothesis is workload mix (openai-gpt is FFN/GEMM-dominated
> with clean contractions; gpt-neox has a tiny-K QK^T + GELU mix). **WP-T0 measures
> this before any kernel work** (§7). Do not size kernels against an unverified split.

---

## 2. Locked decisions

| # | Decision | Choice | Consequence |
|---|----------|--------|-------------|
| 2.1 | **Target** | **Beat O3 across the board, attention included** | the *ambition* (user-chosen over the safer "beat-FFN / parity-attention" option). The tiny-K outer-product kernel **and** cache-tiling-on are both *mandatory to build*. **Honest reconciliation:** §4.2 shows tiny-K QK^T is BW-bound, so beating O3 there may hit a memory-traffic ceiling. T4 is the spike that *tests* whether the ambition is physically reachable; if the ceiling blocks it, that is a **finding reported back for re-decision** (§9.1), not a silent walk-back. The target stands; the spike tells us if it's attainable. |
| 2.2 | **Scope** | **Dense contractions + bias/GELU epilogue fusion** | cover QK^T, FFN up/down, QKV/output projections, attn·V as matmuls; fold bias/GELU into the GEMM nest. **No flash-style softmax fusion** (softmax stays a downstream loop). The epilogue-fusion *capability* is committed (T6 builds it); whether it *fires* on a given epilogue is roofline-cost-gated like every other decision (§5.4). This consciously claims the epilogue-fusion territory that WP-O3 deferred — *for the GEMM case only*. |
| 2.3 | **Injection point** | **Affine level — extend `affine-register-block`** | keep the decoupled textual-MLIR architecture (no onnx-mlir krnl dialect/version dependency, per CLAUDE.md); reuse demote/register-block/promote; stay general for PolyBench/Polygeist. We reconstruct contraction structure via demote rather than reading `krnl.matmul`. |
| 2.4 | **Execution regime** | **Static prefill only (seq=128, M ≥ mr)** | tile sizes are compile-time constants against known M/N/K; no runtime guards. **Decode/GEMV (M=1) and dynamic seq-len are named deferred WPs** (§8). |

Derived technical decisions (not user-facing, settled by analysis + prior verdicts):

- **Packing stays Phase-deferred + spike-gated** (§5.5). The prior `OPERAND_PACKING_FINDINGS.md`
  no-pack verdict was measured on cache-resident BLIS-3 shapes; it does **not** auto-transfer
  to large-FFN (K ≥ 512), but cache-tiling captures the win first (+2.5–3 % packing residual).
  Turn cache-tiling ON first; only spike packing if we still trail O3 on FFN.
- **MachineModel is made first-class by refactor-then-extend** (§4, Q8 default): extract the
  existing inline tile logic into named, lit-tested queries (byte-identical), then layer the
  new compute-roofline arm + kernel-kind selection behind a `hasExplicitGemmModel` gate.

---

## 3. Problem analysis — why transformer GEMMs miss today

Three independent gates, verified against `lib/Transforms/AffineRegisterBlock.cpp`
(1295 lines) and `include/drcompiler/Transforms/Passes.td`.

### 3.1 Gate (1): iter_args vs memref accumulator form — **already solved**

onnx-mlir emits every contraction as an SSA scalar reduction:
`%r = affine.for %k iter_args(%a=%c0){ mulf; addf; affine.yield } ; affine.store %r, %C[i,j]`.
The k-loop body holds no load/store pair, so `collectAccumulators`
(`AffineRegisterBlock.cpp:173`) returns empty and Stage-2 discovery (`:1100`) skips it.
**`dr-scalar-reduction-demote` (`Passes.td:211`) already rewrites iter_args → memref
accumulator form** so the band becomes a perfect reduction nest the matcher accepts.
This gate is *not* a transformer gap; the spec inherits the fix. The 20 s naive lowering
is what survives when, *after demote*, gates (2)/(3) still reject or under-serve the shape.

### 3.2 Gate (2): the two-spatial-loop requirement — hard-drops M=1 GEMV

Stage-2 (`AffineRegisterBlock.cpp:1100-1109`) requires both an inner spatial loop `sIn`
(indexes the accumulator) and `sOut = enclosingSpatial(sIn, store)` (`:1109`).
`enclosingSpatial` (`:394`) returns `nullptr` if no enclosing loop indexes the store
address. A decode-phase GEMV (`M=1`, nest `j–k`, accumulator `C[j]`) has a single spatial
loop `j` ⇒ `sOut = null` ⇒ band rejected ⇒ naive scalar lowering. **This is the structural
reason the autoregressive/decode path is uncovered.** Deferred (§8, decision 2.4) but
named here because it is a *structural* miss, not a quality miss.

### 3.3 Gate (3): tiny-K / tiny-M jam degeneration — the prefill killer

Shapes that pass (1)+(2) still emit bad code because the fixed `mr=8 × nr=16` unroll-jam
(`Passes.td:620,622`; jam at `AffineRegisterBlock.cpp:1182,1218`) is wrong for transformer
dimensions:

- **QK^T `[B=4, 128, 128, K=8]`** — K=8 reduction trip. After the 8×16 jam the inner
  reduction is 8 iterations; broadcast vectorization over `vl=8` on a K=8 loop is one
  vector step. The micro-kernel amortizes nothing → "fires but ~no-op." This is the
  **39 %-FLOP** shape in the gpt-neox census.
- **Batched outer dim `b=4` jammed by `mr=8`** → `loopUnrollJamByFactor` with trip 4 < factor 8
  fully unrolls, then `findReductionLoopUnder` re-finds under `i` (`:1186`) — **works today**,
  guarded by that re-find (a WP-O2 conv-band regression fix); "fragile" = it leans on
  full-unroll-then-re-find rather than handling the short trip directly. **T3 must add a lit test
  pinning this batched path** before enabling transformers on it.
- **Large FFN `[128, 3072, 768]`** would benefit from cache-blocking, but the in-pass
  mc/nc/kc shrink loop (`AffineRegisterBlock.cpp:957-1050`, already using `effectiveLLC`) is
  **gated OFF by default** (`cacheTile`, `Passes.td:624` default `false`). The proven BLIS-scale
  1.76–2.48× cache-tiling win is currently **unrealized on the default path.**

### 3.4 The central tension — transformer GEMMs are short-and-fat

| dimension regime | resnet50 (where we win) | transformer prefill (where we lose) |
|---|---|---|
| K (reduction) | large (1×1 conv: C_in 64–2048) | **tiny** (gpt-neox: 8–37; FFN in larger models: 768–3072) |
| M·N (spatial) | moderate | **large** (128 × {128,768,3072}) |
| kernel that fits | broadcast GEMM micro-kernel, deep-K amortization | **outer-product** kernel: jam M & N, hold tiny-K reduction in registers |

The register-block micro-kernel was tuned for the *deep-K* regime. The dominant transformer
prefill shape is the *opposite*. A single kernel family cannot serve both — hence the
**kernel-kind selection** in §5.

### 3.5 Shape catalog (gpt-neox prefill census, all dims static)

| Shape (B,M,N,K) | role | FLOP share | gate that bites | coverage today |
|---|---|---|---|---|
| (4,128,128,**8**) | QK^T attention scores | **39 %** | (3) tiny-K | weak (fires, no amort.) |
| (1,128,96,32) | FFN up-project | 29 % | (3) small-K | partial |
| (1,128,32,37) | GELU erf-basis up* | 11 % | (3) | partial |
| (1,128,37,32) | GELU erf-basis down* | 11 % | (3) | partial |
| (1,128,32,32) | output projection | 10 % | (3) | partial |
| (4-level) | attn-scores · V | — | batched, N=8 tiny | weak |
| **M=1 GEMV** | decode FFN/QKV | n/a in *prefill* census (but ≈dominant in autoregressive *decode* latency) | **(2) hard miss** | **none** — WP-T7, blocked on a decode harness (none today) |

\* The two "GELU erf-basis" shapes are a Polygeist erf-approximation artifact (rank-37 basis),
not a generic transformer op; they vanish under a different GELU lowering. Treat as
incidental, not a primary kernel target.

---

## 4. The MachineModel as a first-class resident (core of the ask)

Today tiling/blocking decisions are **scattered and partly hardcoded**: `mr`/`nr` are pass
options (`Passes.td:620,622`), `vl` is derived from `MachineModel::preferredVectorElems`
(`MachineModel.h:209`, called at `AffineRegisterBlock.cpp:799`), and cache tiles are an
**inline manual halving loop** (`AffineRegisterBlock.cpp:1024-1050`) that is not reusable or
testable. The roofline has a **bandwidth arm** (`streamCycles`, `MachineModel.h:166`) but
**no compute arm** — so the model cannot tell whether a tiny-K GEMM is compute- or BW-bound,
which is exactly the QK^T decision.

Making the MachineModel first-class means: **one query owns the GEMM blocking decision**, the
roofline is **two-armed**, and every knob is **JSON-gated for byte-identity**.

### 4.1 New gating flag (mirrors `hasExplicitVectorModel` at `MachineModel.h:98`)

```cpp
bool hasExplicitGemmModel = false;   // set true iff JSON `gemm` block present
```
Set in `MachineModel::fromJson` when **either** a `gemm` object **or** `arch.fmaUnits` appears
(fmaUnits-alone enables roofline kernel-kind dispatch without overriding tile sizes; a full
`gemm` block overrides tiling too).

**Call semantics — the byte-identity contract.** `affine-register-block` calls `gemmBlocking()`
**only when `hasExplicitGemmModel == true`.** When false, the pass keeps its *exact current path*:
hardcoded `--mr=8/--nr=16` options (`Passes.td:620,622`), `vl` derived via `preferredVectorElems`
**only if `hasExplicitVectorModel`** (today's behavior, unchanged), `cacheTile` off (`Passes.td:624`).
So `gemmBlocking` is a *second* code path, not a replacement — it cannot perturb the default. Even
when true, explicit CLI `--mr/--nr/--vl/--cache-tile` still override the configurator (it fills only
what the user did not pin). This is the non-negotiable regression gate, checked by two tests (§6.5):
(a) no JSON ⇒ IR bit-identical to pre-T1; (b) JSON with `arch.fmaUnits=2` but no tiling block ⇒
`gemmBlocking` returns the default tiling (kernel-kind dispatch active, tile sizes unchanged).

### 4.2 Compute roofline arm (the missing half)

New `arch` JSON field (in `CpuArchJsonParams`, `CpuCostModel.h:33`):
```cpp
std::optional<unsigned> fmaUnits;   // FP FMA pipes (Zen4 = 2; default keeps compute arm dormant)
```
New MachineModel members:
```cpp
// flops/cycle at the throughput-effective native vector width.
//   = 2 (FMA) * (vectorBitsNative/ (elemBytes*8)) * fmaUnits
double peakFlopsPerCycle(int64_t elemBytes) const;

// compute arm of the roofline (pairs with streamCycles, the BW arm).
double computeCycles(int64_t flops, int64_t elemBytes) const;

// ridge-point arithmetic intensity (flops/byte) = peakFlops / bytesPerCycle.
// A GEMM whose AI is below this is BW-bound; above it, compute-bound.
double ridgeIntensity(int64_t elemBytes, bool fromDRAM) const;
```
`ridgeIntensity` is what classifies tiny-K QK^T (very low AI → BW-bound, so the win is in
**reducing memory traffic / increasing M·N reuse**, not deepening the reduction) vs deep-K
FFN (compute-bound → register-blocking amortization is the lever). Without `fmaUnits` set the
compute arm is inert and `gemmBlocking` falls back to the BW-only heuristic == today.

**Contracts (settled at spec level; exact rounding/epsilon are T2 detail):**
- `elemBytes` is in **bytes** (4 for f32); all three methods guard `elemBytes > 0`.
- `peakFlopsPerCycle = 2.0 · (vectorBitsNative / (8·elemBytes)) · fmaUnits` — the 2 is FMA's two
  flops; it uses `vectorBitsNative` (the *throughput* width, per the `preferredVectorElems`
  convention at `MachineModel.h:209`), **not** `vectorBitsArch`.
- `ridgeIntensity(elemBytes, fromDRAM) = peakFlopsPerCycle / bytesPerCycle(fromDRAM)`, where
  `bytesPerCycle` is **the same bandwidth `streamCycles` uses** (`MachineModel.h:166`): DRAM vs LLC
  selected by `fromDRAM`, derated by `activeThreads` in interspersed mode, full in exclusive mode.
  A band is BW-bound iff its arithmetic intensity < `ridgeIntensity`. Binding classification and
  costing to the *same* bandwidth source is by construction — classifying with one and costing with
  another would invert the decision.
- **`fmaUnits` reference (absence ⇒ compute arm inert ⇒ default behavior):** Zen4/Zen5 = 2,
  Intel Xeon SKX/CLX/ICX = 2, ARM Neoverse (SVE) = 2, Apple M = 4, Power = 2. These are *FMA-issue
  pipes*, not lane counts.

### 4.3 The GEMM configurator — single first-class entry point

```cpp
enum class GemmKernel { Broadcast,    // deep-K: today's mr×nr broadcast micro-kernel
                        OuterProduct, // tiny-K: jam M&N, full-unroll K in registers
                        Gemv };       // M=1 (deferred WP-T7)

struct GemmTiling {
  unsigned mr, nr;        // register micro-tile
  unsigned vl;            // vector lanes
  int64_t  kc;            // L1 K-panel
  int64_t  mc, nc;        // L2/LLC macro-tile
  GemmKernel kind;
  bool      cacheTile;    // whether to emit the mc×nc×kc cache loops
};

// THE query. Picks kernel kind by AI/K, sizes the register tile by the register
// budget, kc by the L1 panel fit, mc/nc by effective L2/LLC. Replaces the inline
// halving loop + scattered vl derivation. Pure function of the machine + (M,N,K).
GemmTiling gemmBlocking(int64_t M, int64_t N, int64_t K, int64_t elemBytes) const;
```

`gemmBlocking` composes the supporting queries below. **`affine-register-block` calls it
once per discovered GEMM band** at the family-select stage (replacing the hardcoded mr/nr +
separate vl + default-off cacheTile). When `hasExplicitGemmModel == false`, it returns
`{mr=8, nr=16, vl=preferredVectorElems(eb,8,16), kc=K, mc=M, nc=N, Broadcast, cacheTile=false}`
— today's behavior exactly.

### 4.4 Supporting queries (extract existing logic → named, testable)

```cpp
// extracted from preferredVectorElems' accumulator-fit loop (MachineModel.h:209-226):
bool canFitAccumulators(unsigned mr, unsigned nr, unsigned vl, int64_t elemBytes) const;

// kc s.t. A-panel(mr×kc) + B-panel(kc×nr) + C-tile(mr×nr) fit effective L1.
int64_t maxL1Kc(unsigned mr, unsigned nr, int64_t elemBytes) const;

// {mc,nc} s.t. (mc*kc + kc*nc + mc*nc)*eb fits effective L2 (checked vs effectiveLLC).
// Promotes the inline shrink loop at AffineRegisterBlock.cpp:1024-1050 verbatim.
std::pair<int64_t,int64_t> l2Tile(int64_t M, int64_t N, int64_t kc, int64_t elemBytes) const;

// bridge CpuRegisterJsonParams (gp/fp/vec/predBudget, CpuCostModel.h:66) into the model.
struct RegisterBudget { unsigned gp, fp, vec, pred; };
RegisterBudget registerBudgets() const;   // today these are parsed but never reach MachineModel
```

`registerBudgets()` closes a real gap: `CpuRegisterJsonParams` (gp/fp/vec/predBudget,
`CpuCostModel.h:66`) is parsed and exposed via `CpuCostModel::registerParams()`
(`CpuCostModel.h:112`) but **never bridged into MachineModel**, so the graph-coloring budget and
the vector-tile budget are two disconnected numbers. It is a **pure read** of the parsed JSON
(no analysis call). §6.2.

**The 24-vs-32 reconciliation (CROSSCUTTING).** The vector-tile budget `vecRegBudget`
(`MachineModel.h:79`, default 24) is the *full* architectural count (32 zmm) **minus** registers
reserved for the streaming B-panel + broadcast temporaries (≈8 on Zen4). Rule: **register-tile
sizing uses `vecRegBudget` (24); `RegisterPressureAnalysis` uses the full `registerBudgets().vec`
(32).** One number, two derived views — not two independent constants.

**Extraction sources (makes T1 a refactor, not a rewrite — byte-identity provable):**
- `canFitAccumulators` ← the accumulator-fit loop inside `preferredVectorElems`
  (`MachineModel.h:209-226`): `mr·⌈nr/vl⌉ ≤ vecRegBudget`.
- `l2Tile` ← the inline mc/nc/kc halving loop verbatim (`AffineRegisterBlock.cpp:1024-1050`); it can
  return a tile that does **not** fit (the "can't shrink further" fallthrough) — the **caller** then
  decides tile-anyway vs reject the band.
- `maxL1Kc` ← new (no L1-panel concept exists today), analogous fit: largest `kc` with
  `(mr·kc + kc·nr + mr·nr)·eb ≤ effectiveCache(L1)`, assuming A-panel, B-panel and C-tile all
  resident (conservative). Returned **unrounded**; the caller rounds `kc` down to a multiple of `vl`
  when it needs stride-1 vectorized B-panel loads.

### 4.5 Effective cache under SMT/LLC sharing & sharding

`gemmBlocking` sizes mc/nc/kc against `effectiveCache(level)` / `effectivePrivateCache`
(`MachineModel.h:141-156`), **not** raw sizes — so the tile already respects SMT siblings and
LLC co-tenants (the contention model from `CROSSCUTTING.md`). **T1–T6 size tiles for
single-threaded execution** (`thread.activeThreads = 1` default) — no `ThreadModel` parameter is
added now. When a (future) `dr-shard` parallelizes a GEMM, it will re-invoke `gemmBlocking` with an
updated `ThreadModel` so the per-thread macro-tile fits `effectivePrivateCache(L2)` ÷ active
siblings, not full L2 (§6.4). The `ThreadModel` coupling lands with `dr-shard`, **not** in this
spec's WPs — avoiding an unused, untestable API surface.

---

## 5. Pass design — affine-level kernel families

`affine-register-block` already runs a Stage-1/2 discovery + per-band family select. We add a
**kernel-kind dispatch** keyed on `gemmBlocking().kind`, and two new kernels. No new pass; no
new dialect dependency (decision 2.3).

### 5.1 Discovery — unchanged for prefill

Stage-2 (`:1100`) already finds the `(sOut, sIn, kLoop)` triple for two-spatial-loop GEMMs.
For each discovered band, compute static `(M,N,K)` from the loop extents and call
`gemmBlocking(M,N,K,elemBytes)`.

### 5.2 Broadcast kernel (deep-K, today's path) — now machine-sized

The existing broadcast micro-kernel stays, but `mr/nr/vl/kc/mc/nc` come from `gemmBlocking`
instead of options/inline-loop. Deep-K FFN (K ≥ 512) now gets **cache-tiling ON** via
`tiling.cacheTile=true` from the configurator (the proven 1.76–2.48× lever, §3.3).

### 5.3 Outer-product kernel (tiny-K, NEW) — the QK^T 39 % shape

`gemmBlocking` returns `OuterProduct` **only** for `K ≤ threshold` — which is exactly what makes
the full unroll safe (large K always routes to `Broadcast`, never fully unrolled). The threshold is
the `K` at which `ridgeIntensity` says the reduction is long enough to amortize a broadcast
micro-kernel, floored at the compiler's practical unroll limit (~16). When `kind == OuterProduct`:
emit an **outer-product accumulation** — jam M (`mr`) and N
(`nr/vl`) into a register tile, **fully unroll the K reduction** (no inner reduction loop),
accumulate `mr × ⌈nr/vl⌉` vector registers across the K rank-1 updates. This is the
structurally correct kernel for short-and-fat GEMMs: the spatial M·N dimension carries the
parallelism, the tiny K is consumed in registers. **Spike-gated (WP-T4):** flag explicitly
that this is unmeasured and may yield no win over O3 on K=8 — the BW-bound classification
(§4.2) says the ceiling is memory traffic, so the realistic target is *parity-to-modest-win*,
not a multiple.

### 5.4 Epilogue fusion (bias/GELU, NEW) — decision 2.2

After the C-tile is computed in registers (before the `affine.store`), fold a trailing
elementwise bias-add and/or GELU that reads the same C index and has no other consumer. This
saves a full C round-trip to memory + reload (the BW-bound regime cares most). Detect by
pattern: an immediately-following affine loop over the same `(i,j)` whose only input is `C`.
Cost it with the **two-armed roofline** — fuse iff the saved C-traffic cycles
(`streamCycles`) exceed the extra register pressure cost. Stays inside the GEMM nest; **does
not** pull in softmax (decision 2.2). This claims the WP-O3-deferred epilogue territory for
the GEMM case only — document the boundary so the general eltwise-fusion WP doesn't double-cover.

### 5.5 Packing — deferred, spike-gated (NOT in the default plan)

Per §2 derived decision: cache-tiling first. A packing spike (WP-T9) runs **only if**
cache-tiling-on still trails O3 on large-FFN. If it runs, packing copies the B-panel
(`kc×nc`) to a contiguous buffer to convert strided column access into stride-1 streaming;
gate the emit on `gemmBlocking` returning a `pack` hint costed by `streamCycles` of the copy
vs the strided-reload penalty. Do **not** reverse the no-pack verdict speculatively.

---

## 6. Cross-cutting impact

### 6.1 Pipeline order (fixed — do not reorder)

`demote → (distribute / time-tile) → affine-register-block → eltwise-fuse → loop-tile`. The
new kernel-kind dispatch and epilogue fusion live **inside** `affine-register-block` at the
family-select/epilogue stage; nothing new is inserted between passes. Demote still produces
the perfect reduction nest; promote (`Passes.td:248`) still raises sub-VL leftovers back to
iter_args. Tiny-K bands the outer-product kernel takes need **no special promote handling**: the
kernel emits the K reduction **fully unrolled**, so no `affine.for %k` loop remains in the IR —
`dr-scalar-reduction-promote` (which raises *leftover loops*) finds nothing to raise. The absence
of the loop **is** the exclusion mechanism; no IR marker is needed. A lit test pins this: tiny-K
output contains no `affine.for %k`, only inlined rank-1 updates.

### 6.2 Register-pressure analysis (CROSSCUTTING unification)

`CROSSCUTTING.md` flags a `regblock ↔ RegisterPressureAnalysis` merge and a register-budget
24-vs-16/32 reconciliation. The outer-product kernel raises register pressure (it holds the
whole `mr × ⌈nr/vl⌉` C-tile **plus** A/B operands across the unrolled K). `gemmBlocking`'s
`canFitAccumulators` uses `vecRegBudget` (24) for the C-tile only; the **full** kernel budget
(C + A-broadcast + B regs + address regs) must route through `registerBudgets()` and, where it
exists, `RegisterPressureAnalysis`'s graph-coloring result — otherwise tiny-K over-jam spills.

**Control flow (no circular dependency).** `registerBudgets()` is a pure JSON read; it does **not**
call `RegisterPressureAnalysis`. The outer-product selection in T4 is *tentative*: `gemmBlocking`
pre-checks a coarse register estimate (C-tile + A-broadcast + B + address ≤ `registerBudgets().vec`)
before returning `OuterProduct`; the later, separate `RegisterPressureAnalysis` pass may still flag
a spill, in which case register-block emits a diagnostic and **falls back to `Broadcast`**. This
spec **requires `registerBudgets()` to land in T1** (a T1 exit criterion, §7) so the tile budget and
the coloring budget are one number *before* T4 builds the pressure-sensitive kernel.

### 6.3 Reuse analysis & the other cost-model consumers

`MemoryFission`, `AffineLoopTilingCostModel`, `AffineLoopFusionCostModel`, `AffineStencilTimeTile`
and `DataRecomputation` are the other `MachineModel` consumers. **T1 audits exactly which fields
each reads** (the mapping phase did not verify this) before `gemmBlocking` claims to be the
**shared GEMM-tile authority** that keeps fission and the GEMM pass from disagreeing on what fits
L2. `l2Tile` must be consistent with `ReuseAnalysis::loopCarriesEvictedReuse` (the
cacheLineBytes-aware reuse-distance
test): a tile `gemmBlocking` declares L2-resident must not be one `ReuseAnalysis` flags as
evicting. Add `test/AffineRegisterBlock/gemmblocking-looptile-consistency.mlir` (a large FFN shape,
e.g. 128×768×3072 f32) asserting `gemmBlocking` and the loop-tiling cost model pick the **same** L2
macro-tile — do not land T1 without it.

### 6.4 Parallel codegen / `dr-shard` (PARALLEL_CODEGEN_SPEC)

drcompiler emits **serial** code; the cost model *decides*, the future `dr-shard` *executes*
parallelism (decision inherited from `CROSSCUTTING.md` + `PARALLEL_CODEGEN_SPEC.md`). The GEMM
spec must **not** emit OpenMP. But it must be *shard-ready*: the `mc × nc` macro-tile is the
natural shard granularity, so `gemmBlocking`'s output is exactly what `decideShard()` consumes,
and the two-armed roofline answers "shard or stay serial?" (don't shard a BW-bound tiny-K
kernel that's already saturating the bus; do shard a compute-bound deep-K FFN). When sharded,
the macro-tile is sized against `effectivePrivateCache` ÷ active siblings (§4.5). This keeps
the transformer GEMM work and the parallel-codegen work on **one** machine model.

### 6.5 Byte-identity & test discipline (non-negotiable)

- Default JSON (`hasExplicit*Model == false`) ⇒ every existing PolyBench/resnet50 result and
  the full lit suite stay **bit-identical**. Each WP adds a lit test proving its new behavior
  fires **only** under an explicit `gemm`/`arch.fmaUnits` JSON, and a companion test proving the
  default path is unchanged.
- **T1**: full `ninja -C build check-drcompiler` passes **bit-identical** (not just resnet50/PolyBench
  spot-checks); `registerBudgets()` reachable from `gemmBlocking`.
- **T6**: a lit test proves epilogue fusion **fires** when the roofline favors it and **does not**
  fire otherwise, plus a default-JSON companion proving no fusion without the gemm model.
- One local commit per WP; **never push**; spike-first; report measurement before implementing
  the automated decision.

---

## 7. Work packages (each measurement-gated)

| WP | Title | Gate / exit criterion | Risk |
|----|-------|----------------------|------|
| **T0** | **Profile the gap** | **method: onnx-mlir `--instrument-stage=Onnx --InstrumentReportTime`** (per-ONNX-op runtime timing — verified available) on openai-gpt **and** gpt-neox, naive vs o3; attribute the gap to op types (MatMul/Gemm vs rest). Interpretation: per-op (or whole-graph) naive/o3 ratio >20× ⇒ deep-K dominates (prioritise T5); <5× ⇒ tiny-K (prioritise T4). Write `ONNX_TRANSFORMER_RESULTS.md` before T4/T5. | low — pure measurement; **blocks all kernel work** |
| **T1** | MachineModel refactor (first-class, byte-identical) | extract `canFitAccumulators`/`maxL1Kc`/`l2Tile` from inline logic **and bridge `registerBudgets()` into MachineModel** (required by T4); `gemmBlocking` called only when `hasExplicitGemmModel`; audit other cost-model consumers (§6.3); **full `check-drcompiler` bit-identical** + zero-IR-change lit on resnet50 + PolyBench | low |
| **T2** | Compute roofline arm | `fmaUnits` JSON + `peakFlopsPerCycle`/`computeCycles`/`ridgeIntensity`; lit test ridge-point classification (tiny-K → BW-bound, deep-K → compute-bound); inert without `fmaUnits` | low |
| **T3** | `gemmBlocking` drives register-block | route mr/nr/vl/cacheTile through the configurator; kernel-kind dispatch scaffold; **byte-identical at default JSON** | med |
| **T4** | Tiny-K outer-product kernel | spike first (hand-write QK^T K=8 kernel, measure vs o3); automate iff spike ≥ parity. Honest possible outcome: **no win** → document and stop | **high** |
| **T5** | Cache-tiling ON for deep-K | flip `cacheTile` via `gemmBlocking` for K ≥ threshold under gemm model; realize the 1.76–2.48× BLIS lever; measure FFN-heavy model | med |
| **T6** | Bias/GELU epilogue fusion | fuse trailing eltwise into the GEMM C-write; roofline-costed; measure traffic reduction; **lit test: fires when roofline favors, aborts otherwise, no-fuse at default JSON** | med |
| **T7** *(deferred)* | Decode/GEMV (M=1) gate-2 fix | new single-spatial-loop recognizer; **needs a decode benchmark first** (none today) | high |
| **T8** *(deferred)* | Dynamic seq-len | runtime tile guards + symbolic tiles; ~2× codegen complexity | high |
| **T9** *(deferred, spike)* | Packing for large FFN | trigger: T5+T6 (cache-tiled + fused) **still** trail O3 on a large-FFN model; spike-measure real FFN shapes before any code | med |

Critical path to a verdict: **T0 → T1 → T2 → T3 → {T4 ∥ T5} → T6**, then a **measurement gate** —
if T5+T6 still trail O3 on the FFN-dominant model, spike T9 (packing); if the target is met, stop.
T4 and T5 serve *different* models (tiny-K vs deep-K), so T0 picks which is on the critical path first.

---

## 8. Out of scope / deferred (explicit)

- **Decode/GEMV (M=1)** — structural gate-2 miss; deferred (decision 2.4, WP-T7). Named, not forgotten.
- **Dynamic seq-len** — deferred (decision 2.4, WP-T8).
- **Flash-style attention fusion (softmax in-loop)** — out (decision 2.2). Softmax stays a downstream loop; we only beat O3 on attention via better tiny-K *matmul* kernels.
- **Operand packing** — not in the default plan; spike-gated WP-T9 behind cache-tiling-on.
- **krnl.matmul dialect hook** — rejected (decision 2.3). Revisit only if a spike shows demote-reconstruction is itself the bottleneck.
- **General eltwise/BN fusion (WP-O3)** — remains separately deferred. Boundary with T6: **T6 fuses only eltwise that immediately follows a GEMM and has no other consumer**; WP-O3 covers everything else and **must check** (IR pattern) that an epilogue wasn't already folded into its parent GEMM by T6 before fusing it again — no epilogue fused twice.

---

## 9. Risks & honest unknowns

1. **Tiny-K may have no win (highest).** QK^T K=8 is BW-bound (§4.2); O3 may already be at the
   memory ceiling. WP-T4 is spike-gated precisely so we can *stop* and report "parity, no win"
   without sunk-cost automation. The 39 % FLOP share means a tiny-K *loss* caps the whole result —
   so T0 must confirm tiny-K actually dominates the runtime gap before T4 is funded.
2. **The 36× split is unverified** (§1 flag) — T0 is mandatory and first.
3. **Byte-identity drift during the T1 refactor** — extracting buried logic risks perturbing the
   default path; mitigated by the zero-IR-change lit test as the T1 exit criterion.
4. **Register over-jam on the outer-product kernel** — needs the unified register budget (§6.2);
   without `registerBudgets()` bridged, tiny-K jam will spill and erase the win.
5. **gpt-neox is tiny-K, larger models are deep-K** — the two kernels (T4 vs T5) serve different
   models; T0's per-model breakdown decides which to build first. Do not assume the benchmark's
   shape distribution generalizes.

---

## 10. Definition of done

- T0–T3 + T6 landed; **both T4 and T5 are built** (per §2.1), and **at least the one T0 marks
  gap-dominant lands with a measured result** (win *or* honest no-win). The other lands too unless
  its spike returns an honest no-win — which is itself an acceptable, reported outcome (§9.1), not a
  campaign failure.
- `MachineModel::gemmBlocking` is the single authority for GEMM tiling, two-armed roofline live,
  all new behavior JSON-gated, default path bit-identical, lit suite green.
- End-to-end re-run of `scripts/onnx-codegen-bench.sh` on all four models with the verdict table
  updated; the transformer rows show the new number (parity or win) with correctness preserved.
