# TRANSFORMER_ELTWISE_SPEC — vectorize the attention/eltwise tail

**Status:** Scope (2026-06-16). Follow-up to `TRANSFORMER_KRNL_SPEC.md` (T1–T5c
landed: GEMMs now beat onnx-mlir --O3 on openai-gpt 1.6× / resnet50 1.39×). The
**one place we still trail o3 is gpt-neox** (1.59× vs 2.27×), and the WP-T4 spike
proved why: it is **not** the GEMMs (those are fully vectorized) — it is the
**scalar attention/eltwise tail** (softmax, masks, activations). This is the
concrete, measurement-justified instantiation of the long-deferred **WP-O3**
(general eltwise/BN fusion+vectorization), scoped to what the transformers
actually need.

Default OFF / byte-identical until a WP lands behind its gate, same discipline as
the GEMM campaign.

---

## ⚠ E0 FINDING (2026-06-16) — the vectorization premise is WRONG; the lever is FUSION

The E0 spike (measured, `/var/tmp/cmp-gn-t5c/`) overturns this spec's lead
(§5.1–5.3 bounded vectorization).  **gpt-neox's tail gap to o3 is NOT
vectorization-addressable:**

- The scalar `exp` *does* vectorize via the backend (`clang -fveclib=libmvec`:
  5 scalar `@llvm.exp.f32` → 80 vector `_ZGV…exp`) — but **e2e: NO win** (4.04 ms
  vs 3.86 ms baseline). So the exp was never the bottleneck (the §1 "0.4 ms"
  estimate was wrong).
- **Max** backend vectorization (`clang -O3 -march=native -ffast-math
  -fveclib=libmvec`) gives **no speedup** (4.13 ms; more vector ops, same time) —
  the classic **memory-bound** signature: clang already vectorizes the
  vectorizable tail, and adding compute throughput does nothing.
- Cause: gpt-neox attention is **traffic-bound** — 284 materialized buffers, 210
  `[4,128,128]` score-matrix round-trips (QK^T → scale → mask → softmax(×passes)
  → ·V, each a full read+write of the 256 KB scores).  o3 is faster (2.87 vs
  3.89 ms) because it **fuses** that chain, not because it vectorizes it.

**Consequence — re-prioritize:** E2/E3/E4 (pointwise/row-reduction/transcendental
vectorization) are a **measured no-go for gpt-neox** (clang already does it; the
kernel is memory-bound).  The ONLY lever is **traffic reduction = fusion**:
- **E5** (local softmax/LayerNorm pass-fusion, 3 reads → 1, + fuse mask/scale into
  the softmax read) — the cheaper fusion, cuts some round-trips.
- **E6** (flash-style QK^T→softmax→·V fusion, no score materialization) — the big
  one; reverses GEMM-spec decision 2.2.

### E6.0 flash spike (2026-06-16, `/tmp/flash_spike.c`)

Standalone C, gpt-neox attention shape [H=4, S=128, D=8], naive-materialized vs
flash (tiled online softmax), `clang -O2 -march=native -ffast-math`:
**flash 1.44× faster** (251 → 175 µs), correctness err 1.7e-7. Confirms flash is
a real, correct win on the attention chain. BUT e2e math: ~76 µs/block × 5 layers
≈ **0.38 ms** — closes ~40% of gpt-neox's ~1 ms gap (won't alone beat o3, which is
2.87 vs our 3.89 ms). gpt-neox is the *worst case* (64 KB/head scores fit L2);
flash's payoff scales with seq length (DRAM-spilling scores at seq≥1024 → a
multiple). **Build cost concern:** recognizing the attention chain (QK^T → scale →
mask → 3-pass softmax → ·V across ~10 onnx-mlir-lowered affine loops + reinterpret
buffers) at the affine level (decision 2.3, no krnl dep) is large + fragile +
onnx-mlir-version-specific. Decision pending: full build vs bank the spike.

E2/E3/E4 may still help a *compute-bound* tail on another part/model, but they are
NOT gpt-neox's lever and should not lead.  **Honest scope question for the user:**
flash/fusion (E5→E6) is a large, decision-2.2-reversing effort for a ~1 ms gain on
the *smallest* model, while we already beat o3 on the two representative models
(openai-gpt 1.6×, resnet50 1.39×).  Worth it, or stop here?  (E2/E3/E4 below are
retained for reference but de-prioritized by this finding.)

---

## 1. Motivation — the measured gap (WP-T4 finding)

gpt-neox codegen is **fully vectorized on the GEMM side** (240 `vector.broadcast`,
0 scalar-alloca accumulators; QK^T, scores·V, FFN all vectorized). The residual
gap to o3 is the attention/eltwise tail, which is **entirely scalar** (census of
`/var/tmp/t0/gn.cg.mlir`):

| scalar op (un-vectorized) | count | role |
|---|---|---|
| `math.exp` (scalar) | 5 | softmax exp (≈65K calls × ~30 cyc × heads ≈ **0.4 ms of the ~1.15 ms gap**) |
| `iter_args(f32)` row-reductions | 10 | softmax max-reduce + sum-reduce (over seq=128) |
| `arith.maxnumf` / `arith.divf` | 5 / 38 | softmax max, normalize |
| `arith.select` | 6 | causal mask |
| pointwise copy/transpose/scale loops | many | reshape, scale, mask-add |

The same tail dominates **openai-gpt** beyond its current 1.6× win: 12 `Softmax`,
12 `Gelu`, 24 `LayerNormalization`, 49 `Add` — all lowered to scalar passes. So
this WP **lifts every transformer**, not just gpt-neox.

o3 vectorizes/fuses this tail; we leave it scalar because register-block targets
GEMM reduction bands, not pointwise loops or row-reductions.

---

## 2. Why the tail is scalar today

Three op shapes register-block does not touch:

1. **Pointwise eltwise** — `for…{ C[idx] = f(A[idx], B[idx]) }` (scale, mask-select,
   mask-add, bias, residual, reshape/transpose copies). No reduction → register-block
   skips. Trivially data-parallel; the contiguous inner dim is vectorizable.
2. **Row reductions** — `for row { acc=init; for c { acc = combine(acc, X[row,c]) } }`
   (softmax max & sum, LayerNorm mean & var). A 1-D reduction whose reduction dim is
   the *vectorizable contiguous* dim — the OPPOSITE of a GEMM (vectorize the reduction
   dim, then one horizontal `vector.reduction` at the end), so the GEMM broadcast
   kernel does not apply.
3. **Vector transcendentals** — `math.exp` (softmax), `erf`/`tanh` (GELU). Need a
   vector form (LLVM vector-math intrinsic or a polynomial approx) within the bench's
   1e-4 norm-rel-err.

---

## 3. Scope decisions (inherited from the GEMM campaign)

| # | Decision | Choice |
|---|----------|--------|
| 3.1 | Injection | **Affine level** — a new pass (or register-block sibling) over affine pointwise/reduction loops; no krnl/onnx dialect dependency (consistent with the whole campaign). |
| 3.2 | Machine model | **First-class** — vl from `preferredVectorElems`; a **vector-transcendental cost** in `CpuCostModel`; the **compute roofline arm** (T2) decides compute-bound (exp/erf) vs BW-bound (the copy/scale passes); fusion decisions via the roofline (same currency as fission/T6). |
| 3.3 | Gating | New behavior gated (JSON / a `--vectorize-eltwise` option), **default byte-identical**; lit per behavior + default-unchanged companion. |
| 3.4 | Method | **Spike-first, measurement-gated**; honest no-go allowed. One local commit per WP; never push. |
| 3.5 | Correctness | bench `norm-rel-err ≤ 1e-4` + top-1 is the gate — critical for the vector-exp/erf approximations and for softmax FP reassociation. |

**The one open fork (decide after Phase 1, §6):** bounded vectorization vs
**flash-style fusion**. Decision 2.2 of the GEMM spec ruled flash-attention fusion
OUT *for that spec*; the T4 measurement reopens it as the biggest lever. This spec
**does the bounded vectorization first** (most of the win, low risk) and treats
flash fusion as a separate spike-gated phase — not pre-committed.

---

## 4. Machine-model extensions (keep it first-class)

Mostly reuses what T1/T2 added. New surface:

- **Vector-transcendental cost** — `CpuCostModel` gains a cost for `math.exp`/`erf`/
  `tanh` at scalar vs vector width (a vectorized exp is ~1 vector op replacing `vl`
  scalar calls, but with a throughput, not 1-cycle, cost). Feeds the compute roofline
  arm so `computeCycles` for a softmax/GELU band is real, and the
  vectorize/fuse decision is costed, not assumed.
- **Reduction-kernel selector** — extend `gemmBlocking`'s spirit with a tiny
  `reductionKind(extent, elemBytes)` query: vectorize-the-reduction-dim + horizontal
  `vector.reduction` (row reduction) vs the existing GEMM accumulate. Picks vl and
  whether a tail-peel is needed (extent % vl).
- No new cache surface needed — these passes are streaming; `streamCycles` (BW arm)
  already prices their traffic, and that is exactly what fusion (Phase 2) removes.

Everything stays JSON-gated; absent the eltwise model, the passes are inert.

---

## 5. Pass design (affine level)

### 5.1 Pointwise eltwise vectorizer (Phase 1a)
Match a perfectly-nested pointwise loop whose innermost dim is contiguous (unit
stride in all operands) with no loop-carried dependence; vectorize that dim by vl,
peel the `extent % vl` tail. Covers scale, mask-select (`arith.select` vectorizes),
mask/bias/residual add, reshape/transpose-free copies. Bias/residual that sit on a
GEMM output are the **T6 epilogue-fusion** candidates — coordinate the boundary
(§7).

### 5.2 Row-reduction vectorizer (Phase 1b)
Match `for row { acc=init; for c { acc=combine(acc, X[row,c]) } }` with `combine ∈
{addf, maxnumf, …}` over a contiguous `c`. Emit a `vector<vl>` accumulator across
`c`, a horizontal `vector.reduction` at the end, tail-peel. Covers softmax max &
sum and LayerNorm mean & var. FP reassociation is required (vector reduction
reorders adds) — gated like the GEMM `fastmath` reassoc, validated by the err bound.

### 5.3 Vector transcendentals (Phase 1c)
Lower `math.exp`/`erf`/`tanh` on `vector<vl>` — either via LLVM's vector-math
lowering or an explicit polynomial approx (degree chosen so worst-case rel-err ≪
1e-4). This is the highest-value single piece (the scalar `exp` is ~0.4 ms). Spike
the accuracy first on the bench.

### 5.4 Softmax / LayerNorm as recognized macro-ops (Phase 1d, optional)
Recognize the 3-pass softmax (max → exp+sum → div) and fuse the 3 passes over one
row tile (one read of the row, not three) — a *local* fusion, not flash. Same for
LayerNorm. Cuts the [B,H,S,S] round-trips 3×→1× without crossing into the GEMM.

### 5.5 Flash-style attention fusion (Phase 2 — spike-gated, the fork)
Fuse QK^T → scale → mask → (online) softmax → scores·V over S-tiles kept in
cache/registers, eliminating the [B,H,S,S] scores materialization entirely. Biggest
win, biggest risk (online-softmax numerics, tiling, register pressure with the two
GEMMs + softmax live). **Only if Phase 1 leaves gpt-neox short of o3.** Reverses
GEMM-spec decision 2.2 — an explicit re-decision point with Phase-1 numbers in hand.

---

## 6. Work packages (measurement-gated)

| WP | Title | Gate / exit |
|----|-------|-------------|
| **E0** | Profile the tail | per-pass cost on gpt-neox + openai-gpt (where instrumentation allows, else IR-FLOP census like T0). Confirm softmax/exp dominates; size the ceiling. Blocks E2+. |
| **E1** | Vector-transcendental cost + roofline | `math.exp`/`erf` vector cost in CpuCostModel; feeds computeCycles. Lit: ridge classifies softmax compute-bound. Byte-identical. |
| **E2** | Pointwise eltwise vectorizer (5.1) | vectorize contiguous pointwise loops; lit fires-under-gate / inert-default; bench correctness. |
| **E3** | Row-reduction vectorizer (5.2) | softmax max/sum + LayerNorm mean/var vectorized; FP-reassoc gated; err ≤ 1e-4. |
| **E4** | Vector exp/erf/tanh (5.3) | accuracy spike FIRST (worst-case rel-err); then wire. The 0.4 ms lever. |
| **E5** | Local softmax/LayerNorm pass-fusion (5.4) | 3→1 row reads; measure traffic cut. |
| **E6** *(spike, deferred)* | Flash-style attention fusion (5.5) | only if E2–E5 leave gpt-neox < o3; re-decide 2.2 with data. |

Critical path to a verdict: **E0 → E1 → {E2 ∥ E3 ∥ E4} → E5**, then measure
gpt-neox vs o3; spike E6 only if still short.

---

## 7. Cross-cutting

- **T6 boundary** — bias/GELU sitting directly on a GEMM output are T6 (fuse into
  the GEMM C-write); standalone activations / softmax / LayerNorm are this spec.
  Rule: if the eltwise immediately follows a GEMM with no other consumer → T6;
  else → E2/E5. Neither fuses the same op twice (the WP-O3 handoff rule).
- **T5c interaction** — T5c fissions the bias epilogue into a standalone pointwise
  loop; E2 then vectorizes *that* loop for free. (So T6 becomes "skip the fission,
  fuse instead" — measure which wins.)
- **Cost-model consumers** — the vector-transcendental cost is a CpuCostModel op
  cost; MemoryFission/DR already consume CpuCostModel, so a costed `exp` improves
  their recompute-vs-materialize calls too (a free side-benefit).
- **Parallel codegen** — these passes emit serial vectorized code; `dr-shard`
  (PARALLEL_CODEGEN_SPEC) shards them later. No OpenMP here.
- **Byte-identity** — all gated; default machine + PolyBench + the 223-test lit
  suite stay bit-identical.

---

## 8. Risks & honest unknowns

1. **Vector exp/erf accuracy (highest)** — a polynomial approx must stay ≪ 1e-4
   rel-err across the input range; softmax exp inputs are post-max-subtract (≤ 0),
   which bounds the range and helps. Spike E4 accuracy before wiring.
2. **Softmax FP reassociation** — vector max/sum reorder; max is exact (assoc),
   sum reassociates (gated, err-bounded). 
3. **Headroom realism** — Phase 1 (no fusion) removes the *scalar* cost but keeps
   the [B,H,S,S] round-trips; o3 may also fuse, so Phase 1 might reach parity, not a
   win — E6 (flash) may be needed to *beat* o3 on gpt-neox. Honest: Phase 1's target
   is parity; beating may require the fork.
4. **Generality** — the pointwise/row-reduction vectorizers must not fire on the
   GEMM bands (already handled by register-block) or PolyBench stencils — gate +
   guard like T5c (adversarial review before landing E2/E3).

---

## 9. Definition of done

- E1–E5 landed, gated, default byte-identical, lit green; vector exp/erf accuracy
  proven within 1e-4.
- gpt-neox re-measured vs o3: **parity or better** (E6 spiked only if still short,
  with a recorded re-decision on flash fusion).
- openai-gpt re-measured: its activation/LayerNorm tail vectorized → win extends
  beyond 1.6×.
- The machine model remains the single authority (vl + transcendental cost +
  roofline drive every decision).
