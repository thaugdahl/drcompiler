# ONNX_TRANSFORMER_RESULTS

## ‼ CORRECTED HEADLINE (2026-06-16) — the transformer "wins" were a baseline artifact

The "beat onnx-mlir --O3 by 1.6× on openai-gpt" headline below is RETRACTED. The
`o3` baseline was compiled by onnx-mlir `--EmitObj` with **no target CPU** →
generic **SSE2** (0 AVX, 0 FMA), while our `codegen` used `clang -march=native`
(AVX-512+FMA). Comparing AVX-512 vs SSE2 is an unfair-baseline error. Re-run with
onnx-mlir's fairest achievable target (`--mcpu=znver3`, AVX2+FMA — its bundled
LLVM cannot target Zen4/AVX-512 at all: `--march=native` → "invalid target
znver4"). On Zen4, AVX-512 is double-pumped ≈ AVX2 throughput, so this is a fair
ISA match.

**Fair results (codegen vs onnx-mlir --O3 at its best, AVX2):**

| model | codegen | o3 (fair, AVX2) | verdict |
|-------|---------|-----------------|---------|
| resnet50 (conv) | 0.932 s | 1.242 s | **codegen 1.33× faster** ✅ |
| openai-gpt (transformer) | 0.340 s | **0.244 s** | **o3 1.39× faster — we lose** ❌ |
| gpt-neox (transformer) | 4.02 ms | **2.14 ms** | **o3 1.88× faster — we lose** ❌ |

**Honest bottom line:** fairly compiled, onnx-mlir --O3 BEATS our codegen on both
transformers; we beat it only on the conv-heavy CNN (resnet50). The
`canonicalizeAllocaGemm` pass DOES fix onnx-mlir's --O2 scalar-alloca pathology
(none 20.2s → codegen 0.34s, 59×), but onnx-mlir's own --O3 fixes it BETTER
(0.244s) — so the contribution is "a decoupled post-pass recovers MOST of --O3's
GEMM performance from --O2 output (within ~1.4×), and beats --O3 on conv," NOT
"beats --O3 on transformers." `o3host` is NOT a clean control: convert-krnl-to-
affine degrades onnx-mlir --O3 by ~1.85× (gptneox o3 2.14 → o3host 3.95), so its
earlier "tie" numbers are invalid. Everything below predates this correction.

---


## HEADLINE (WP-T5c landed, commit 084a340, 2026-06-16)

`canonicalizeAllocaGemm` promotes onnx-mlir's scalar-alloca accumulator to the
spatial output C[i,j] and fissions init / k-reduction / bias-epilogue into perfect
register-blockable nests → the existing vectorizer + L2 cache-tiler crush the FFN.
Gated on a GEMM-model JSON (`bench/zen4-gemm.json`); default byte-identical.

| model | none | **codegen (ours, +gemm JSON)** | o3 | vs o3 |
|-------|------|-------------------------------|-----|-------|
| **openai-gpt** | 20.37 s | **0.341 s (59.7×)** | 0.549 s (37.1×) | **1.61× FASTER than o3** ✅ |
| gpt-neox | 5.93 ms | 3.86 ms (1.59×) | 2.71 ms (2.27×) | o3 still ahead (tiny-K) |
| resnet50 | 2.10 s | 0.93 s (2.25×) | 1.29 s | 1.39× faster (unchanged) |

> **⚠ BACKEND-CONFOUND CONTROL (H1, the publishable number).** The "vs o3" column
> above crosses BOTH the transform and the codegen backend (host clang
> -march=native vs onnx-mlir --EmitObj). Adding `o3host` (onnx-mlir --O3 krnl
> through the SAME host backend, no dr-opt) isolates the transform effect:
>
> | model | codegen | o3host (same backend) | **codegen vs o3host** | backend delta (o3/o3host) |
> |-------|---------|----------------------|----------------------|---------------------------|
> | resnet50 | 0.929 | 1.271 | **1.37× faster** ✅ | ~neutral (1.02×) |
> | openai-gpt | 0.340 | 0.388 | **1.14× faster** ✅ | their backend 1.41× slower |
> | gpt-neox | 3.99 | 3.73 | **0.93× (tie/loss)** | their backend 1.35× faster |
>
> The honest transform delta is **1.14–1.37×** (resnet50/openai-gpt win, gpt-neox
> tie), NOT the confounded 1.6×; the backend is a ±40% wildcard. Also: onnx-mlir
> --O3 ALSO fixes the --O2 scalar-alloca pathology (o3host 54× over none), so the
> honest framing is "**decoupled post-pass on --O2 output, matches/beats --O3
> same-backend, no compiler fork**" — not "we fix what --O3 leaves broken".

## WP-T4 (tiny-K outer-product) — MEASURED NO-GO (2026-06-16)

Spike (IR analysis of gptneox codegen under the gemm JSON, `/var/tmp/t0/gn.cg.mlir`)
to decide whether a tiny-K outer-product kernel can close gptneox's residual gap
(codegen 1.59× vs o3 2.27×).  Verdict: **no win available; do not build it.**

- gptneox is **fully vectorized** on the GEMM side: 240 `vector.broadcast`, **0**
  surviving scalar-alloca accumulators. scores·V (k=128) vectorized; FFN
  vectorized via T5c; and the **QK^T (k=8) is already optimally vectorized** —
  onnx-mlir pre-transposes K to `[head, k=8, j=128]`, so the broadcast kernel's K
  load is **contiguous in j**, the k=8 reduction is a `vector<8xf32>` iter_args,
  Q is broadcast, the C-tile stays in registers.  An outer-product kernel cannot
  beat a contiguous, register-resident broadcast kernel — there is no tiny-K GEMM
  headroom (the strided-K worry that motivated T4 doesn't occur; onnx-mlir already
  transposed it away).
- The gptneox gap to o3 is the **scalar attention eltwise**: 10 scalar
  `iter_args(f32)` softmax reductions (max + sum) + scalar `math.exp` / `div` /
  mask-select / scale passes over `[4,128,128]` (the scalar `exp` alone ≈ 65K
  calls × ~30 cyc × 4 heads ≈ 0.4 ms of the ~1.15 ms gap).  o3 vectorizes/fuses
  these; we do not.
- That work is **out of scope**: decision 2.2 ruled out flash-style attention
  (softmax) fusion, and general eltwise vectorization is the deferred WP-O3.

Conclusion (matches the WP-T0 prediction): the tiny-K matmul is not gptneox's
bottleneck; a T4 outer-product kernel would not move the e2e number.  The real
gptneox lever is **vectorizing the attention eltwise / softmax** — a separate,
currently-out-of-scope WP.  No code landed (measurement spike).

---

openai-gpt went from **0.98× (no-op)** to **beating onnx-mlir --O3 by 1.61×**.
Correctness: top1 identical, norm-rel-err over all 98304 outputs = 1.5e-6 < 1e-4.
This closes the deep-K transformer gap. gpt-neox remains o3-favored (tiny-K QK^T,
BW-bound — that's the T4 outer-product territory, secondary per T0).

---

# WP-T0 profiling (2026-06-16)

Decides T4 (tiny-K outer-product) vs T5 (cache-tiling deep-K) priority for
`TRANSFORMER_KRNL_SPEC.md`. **Not committed** (local doc).

## Method note

onnx-mlir runtime instrumentation (`--instrument-stage=Onnx --InstrumentBeforeOp
--InstrumentAfterOp --InstrumentReportTime`, env `ONNX_MLIR_INSTRUMENT_FILE`) is
**not wired in this build** — no instrument ops are emitted at `--EmitMLIR` or
`--EmitObj`, no runtime report. Fell back to whole-graph timing (already in hand)
+ static op-type/FLOP census + a register-block firing probe on the generated
affine IR. This is decisive for the T4-vs-T5 gate.

## Whole-graph gap (from the cross-model bench)

| model | naive | codegen | o3 | gap (naive/o3) | spec rule |
|-------|-------|---------|-----|----------------|-----------|
| openaigpt | 20.25 s | 20.76 s (**0.98×**) | 0.555 s | **36.5×** | >20× ⇒ deep-K ⇒ **T5** |
| gptneox | 5.93 ms | 3.95 ms (1.50×) | 2.68 ms | 2.2× | <5× ⇒ tiny-K ⇒ T4 |

## openaigpt op/FLOP census (the 36× model)

`--EmitONNXIR`: **48 `onnx.Gemm` + 24 `onnx.MatMul`** (+ 12 Softmax, 12 Gelu, 24
LayerNorm, 49 Add). 12 layers, d_model=768, FFN=3072, heads=12, head_dim=64.
Per-layer FLOP estimate:
- FFN up + down (K=768 / 3072): ~1.2 GFLOP — **~50 %**
- QKV + output projections (K=768): ~0.6 GFLOP — **~25 %**
- attention QK^T + scores·V (K=64): ~0.6 GFLOP — ~25 %

⇒ **~75 % deep-K** (K ≥ 768), attention K=64 is moderate (not tiny-K). openaigpt
is a deep-K / cache-tiling workload, **not** a tiny-K one.

## Firing probe (does register-block even run?)

`demote → affine-register-block{mr=8 nr=16} → promote` on the generated affine IR:
- **gptneox**: 240 `vector.broadcast` — fires; gets 1.50× but o3 is 2.21× → the
  **gate-3 tiny-K under-amortization** (K=8 QK^T = 39 % FLOP, fires-but-weak).
- **openaigpt**: 192 `vector.broadcast` — **fires**, yet codegen = 0.98× (≈ none).
  Register-blocked + vectorized but the deep-K FFN Gemms stay **DRAM-bound**: with
  `cache-tile` OFF (`Passes.td:624`) the B-panels (~9 MB) stream from DRAM every
  pass, so register/vector blocking buys nothing on a bandwidth-bound loop.

## Verdict

1. **T5 (cache-tiling ON for deep-K) is THE lever** — it directly addresses the
   openaigpt 36× (the dominant absolute gap), whose Gemms already fire but are
   DRAM-bound. The proven 1.76–2.48× BLIS cache-tiling win applies squarely.
2. **T4 (tiny-K outer-product) is secondary** — it only helps gptneox (small 2.2×
   gap, BW-bound K=8, low expected headroom). Build it (per §2.1) but expect
   parity-to-modest; spike-gate honestly.
3. Implementation order unchanged: **T1 (mm refactor) → T2 (roofline) → T3
   (gemmBlocking dispatch, ensure deep-K fires well) → T5 (cache-tile) → T6**,
   with T4 in the {T4 ∥ T5} slot but lower priority.

Artifacts: `/var/tmp/t0/` (og.affine.mlir 891 MB, gn.*, *.codegen.mlir).

---

# Implementation status (T1–T5, 2026-06-16)

Committed (local, never pushed): `ce8072e` bench harness, `2aafc5d` T1, `d5f8ba5`
T2, `867dc89` T3, `5b81e6c` T5. check-drcompiler 221/0 throughout; default path
byte-identical (all new behavior JSON-gated on hasExplicitGemmModel).

- **T1** MachineModel first-class: `macroTile` (extracted byte-identical),
  `canFitAccumulators`, `maxL1Kc`, `gemmBlocking`, `registers` bridge,
  `hasExplicitGemmModel`.
- **T2** compute roofline arm: `fmaUnits` + `peakFlopsPerCycle`/`computeCycles`/
  `ridgeIntensity` (bound to streamCycles bandwidth).
- **T3** gemmBlocking drives register-block: `if (cacheTile || hasExplicitGemmModel)`,
  per-band decision from gemmBlocking. Lit `gemm-model-autotile.mlir`.
- **T5** model-driven **L2** cache-tiling (budget is effective L2, not LLC —
  openai-gpt's 11 MiB FFN WS fits the 128 MiB V-cache L3, so only L2-tiling
  helps). Mechanism verified on clean GEMMs + gptneox e2e correct.

## BLOCKER for the openai-gpt win (next WP)

The model is wired and fires on perfect-nest GEMMs, but **0 bands tile on
openai-gpt**. Cause (IR-confirmed, `/var/tmp/t0/og.affine.mlir`): onnx-mlir lowers
each FFN `Gemm` to a **scalar-alloca accumulator with an inline bias epilogue**:

```
for i { for j { alloca a; a=0; for k { a += A[i,k]*B[k,j] }; C[i,j] = a + bias[j] } }
```

The j-loop body is not just the k-loop (alloca/init before, bias-add + C-store
after) ⇒ imperfect nest, non-spatial (scalar) accumulator. register-block's
Stage-2 vectorizes it (192 broadcasts) but the cache-tiler's Stage-1b needs a
perfect i-j-k band with a spatially-indexed accumulator, so it rejects it.

**This is the same obstacle as T6**: the inline bias epilogue. The fix unblocks
both:
1. Promote the scalar-alloca accumulator to the C-output (or distribute the
   init + bias epilogue out of the j-loop), yielding a perfect i-j-k reduction
   band that Stage-1b can tile → openai-gpt cache-tiling win.
2. The distributed-out bias is then the clean T6 epilogue-fusion candidate (or
   stays a cheap separate pass over C, 128×3072 ≪ the GEMM).

Next WP (T5b/T3b, then T6): extend Stage-1b band recognition to the
alloca-accumulator + epilogue form. Until then T5 is mechanism-only on this host.

## T5b investigation (2026-06-16) — tiling is NOT the lever; vectorization is

Prototyped a 2D spatial tiler (tile the i->j spatial loops, K-reduction in the
body) since openai-gpt's i->j IS a perfect 2-deep nest even though i-j-k isn't.
It fired -- **72 tiled bands on openai-gpt** (all 48 Gemm + 24 MatMul). But e2e
measured **0.99x (no win)**: codegen 20.64 s vs none 20.33 s, o3 0.55 s (37x).

Decisive cause (synthetic probes, `/tmp/ffn*.mlir`):
- The perfect i-j-k form (no init/epilogue in j) **vectorizes (16 broadcasts) and
  cache-tiles (3 steps)** under the GEMM JSON -- the whole pipeline works on it.
- onnx-mlir's scalar-alloca form gives **0 broadcasts** -- and so does the
  C-indexed form WITH a `C[i,j]=0` init inside j. The init/epilogue statements in
  the j-body make it imperfect, so register-block Stage-2 (enclosingSpatial)
  never vectorizes it.
- The FFN bottleneck is therefore **the scalar dependent accumulation chain**
  (alloca load/add/store round-trip per k-iteration, ~1.5 GFLOP/s, latency-bound),
  NOT B-panel streaming. **Tiling for locality cannot help a latency-bound scalar
  chain** -- only vectorization (vector-register accumulators + parallel lanes)
  can, which needs the perfect register-blockable form.

So the 2D spatial tiler was reverted (no measured benefit standalone). **The real
linchpin is the canonicalization transform:** promote the scalar-alloca
accumulator to a spatial/register accumulator AND fission the init + bias epilogue
out of the j-loop -> perfect i-j-k band -> register-block vectorizes (proven) +
the landed T5 cache-tiling then applies. The existing `distributeLoop` refuses
this (it won't replicate side-effecting init/epilogue stores), so it is a NEW
statement-level fission transform -- essentially reconstructing onnx-mlir's krnl
GEMM canonicalization at the affine level. That is the next WP; it also subsumes
T6 (the bias epilogue becomes the distributed-out epilogue loop / fusion target).
Deferred from this session as a substantial transform with model-level
correctness risk that warrants a dedicated, careful pass (the bench's
norm-rel-err check is the safety net when it is attempted).

