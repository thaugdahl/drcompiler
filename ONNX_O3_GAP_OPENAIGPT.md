# openai-gpt: codegen vs onnx-mlir --O3 — gap analysis

Date: 2026-06-17. Host: AMD Ryzen 9 7950X3D (Zen4, AVX-512), single-thread,
batch 1, seq 128. Model: `openaigpt_Opset18.onnx` (GPT-2: 12 layers, d=768,
heads=12, head_dim=64, FFN=3072). Harness: drcc-benchmarks
`onnx/scripts/onnx-run-bench.sh` (codegen-campaign configs added this session;
see AGENTS.md). All configs verified correct (norm-rel-err vs `none` ≤ 1e-4).

Specimens (code-only IR = baked weights stripped; `.s` = `objdump -d .text`):
`/tmp/cg-ogpt/specimens/{none,codegen,o3host}.{code.mlir,s}` and
`/tmp/cg-ogpt/openaigpt_Opset18/stage_*/`.

---

## 0. Executive summary

There are **two orthogonal gaps**, and they point opposite directions:

| comparison | configs | factor | meaning |
|---|---|---|---|
| **Transform** (same backend) | codegen 0.2865 s vs o3host 0.3721 s | **codegen 1.30× FASTER** ✅ | *our transforms beat onnx-mlir --O3's* |
| **Backend** (same --O3 IR) | o3host 0.3721 s vs o3emitobj 0.2240 s | **EmitObj 1.66× FASTER** | *onnx-mlir's backend beats host-clang* |
| **Net product** | o3emitobj 0.2240 s vs codegen 0.2865 s | **onnx-mlir 1.28× faster** | *backend edge outweighs our transform edge* |

1. **On transforms we already win (1.30×).** Backend-matched (both lowered
   through the same `clang -O2 -march=native` AVX-512), our dr-opt codegen beats
   onnx-mlir --O3's krnl optimizations on openai-gpt. The win is almost entirely
   the GEMMs (96% of matmul FLOP), driven by **register-blocking depth** — see §3.
2. **onnx-mlir's shipping product still wins net (1.28×)** because its native
   `--EmitObj` backend is **1.66× faster than our lowering path on the identical
   --O3 IR**. We proved this is **not** the clang opt-level (`-O2`→`-O3` moves
   codegen 4.3%, o3host 1.3% — §4); it is the `krnl→affine→scf→llvm→clang`
   round-trip our pipeline forces vs onnx-mlir's native `krnl→llvm` lowering.
   This penalty hits *our* codegen too, capping its absolute speed.
3. **The biggest untapped lever is shared and easy on our side:** both configs
   leave **all transcendentals scalar**. Gelu alone (`powf(x,3)` + `tanh` over
   4.72M elements each per inference) is **~20–45% of runtime**, and
   `powf(x,3.0)` is never strength-reduced to `x*x*x` by *either* compiler (§5).

**Bottom line:** we win the algorithm/transform battle; onnx-mlir wins the
backend battle by more. To beat onnx-mlir's product on openai-gpt, attack (a)
the scalar transcendentals (easy, large, we control the MLIR) and (b) the
1.66× lowering-path penalty (hard, highest ceiling). Ranked backlog in §6.

---

### ⮕ PROGRESS UPDATE (2026-06-17) — backlog #1, #2 LANDED; #3 was #2

Two committed steps (local, never pushed; default byte-identical; lit 225/0):

| step | commit | change | codegen median |
|---|---|---|---|
| baseline | — | (canonicalizeAllocaGemm GEMMs) | 286 ms |
| #1 powf→mul | `0c64eff` | `dr-math-strength-reduce` | ~264 ms |
| #2 poly-approx | `aa15241` | vectorize tanh/exp/… (poly) | 239 ms |
| -O3 backend | (harness) | clang -O2→-O3 on call-free IR | **231 ms** |

**Key discovery:** the "1.66× backend gap" (#3) was **not** a lowering/opt-level
mystery — it was the **scalar transcendentals**. onnx-mlir EmitObj emits 0
`tanhf`/`expf`/`powf` (native vector polynomials); our path left 288/96/288 libm
calls. Fixing it at the MLIR level (#2) is the same fix. A backend flag is
insufficient (`libmvec` has no vector `tanh`).

**GEMM polish (#4 vl=16/zmm, #5 single-Kc): both SPIKED → NO-GO** — vl=8 +
model-derived K-tiling is already optimal on Zen4 (§6). **Eltwise super-vectorize:
SPIKED → blocked** (transfer-lowering incompatible with the krnl back-half;
residual ~3% not worth a new pass, §6).

**Final board:** codegen **0.231 s** vs o3host(-O3) ~0.367 (**1.59× win**) vs
o3emitobj 0.224 (**1.03× behind ≈ parity**, was 1.28×). The net gap to onnx-mlir's
best native product is now within the measurement noise floor; the same-backend
transform win is **1.59×**.

---

## 1. Latency board (median of 15, back-to-back, same session)

| config | transform | backend | median | vs none | norm-rel-err |
|---|---|---|---|---|---|
| none | onnx-mlir --O2 | host clang -O2 -march=native | 23.596 s | 1.00× | ref |
| **codegen** | our dr-opt | host clang -O2 -march=native | **0.2865 s** | 82.3× | 8.6e-6 |
| o3host | onnx-mlir --O3 | host clang -O2 -march=native | 0.3721 s | 63.4× | 2.6e-6 |
| o3emitobj | onnx-mlir --O3 | onnx-mlir `--EmitObj` (native) | 0.2240 s | 105× | 2.6e-6 |
| codegen, clang -O3 | our dr-opt | host clang **-O3** | 0.2743 s | — | 8.6e-6 |
| o3host, clang -O3 | onnx-mlir --O3 | host clang **-O3** | 0.3672 s | — | 2.6e-6 |

`codegen` = `func.func(dr-scalar-reduction-demote, affine-register-block{mr=8
nr=16 cpu-cost-model-file=bench/zen4-gemm.json}, dr-scalar-reduction-promote)`.
The GEMM JSON sets `hasExplicitGemmModel` ⇒ `canonicalizeAllocaGemm` +
gemmBlocking + cache-tiling fire (without it, openai-gpt is a no-op at 0.98×).

---

## 2. The input (onnx-mlir --O2, `none`) — what both paths transform

One layer (all 12 identical). Per-layer matmul MAC budget and form:

| kernel | none form | shape | MAC/layer | % matmul |
|---|---|---|---|---|
| FFN-up | scalar-alloca acc, K=768 | 128×3072×768 | 302M | 32% |
| FFN-down | scalar-alloca acc, K=3072 | 128×768×3072 | 302M | 32% |
| QKV-proj | scalar-alloca acc, K=768 | 128×2304×768 | 226M | 24% |
| out-proj | scalar-alloca acc, K=768 | 128×768×768 | 75M | 8% |
| QK^T | iter_args reduce, K=64 | 12×128×128×64 | 12.6M | 1.4% |
| scores·V | iter_args reduce, K=128 | 12×128×64×128 | 12.6M | 1.4% |

Deep-K scalar-alloca GEMMs = **96% of matmul FLOP** (load/add/store the
accumulator to an `alloca` *per k-iteration* — the latency-bound chain that
makes `none` 23.6 s). Plus per-layer scalar transcendentals: 196K `exp`
(softmax), 393K `tanh` + 393K `powf` (Gelu), 256 `sqrt` (2 LayerNorms).

---

## 3. Transform gap (codegen vs o3host, backend-matched): per-kernel

Adversarially verified (12-agent workflow). "winner" = which transform is
faster on the matched backend.

| kernel | % FLOP | winner | root cause (verified) |
|---|---|---|---|
| FFN-up/down | 64% | **codegen** | **accumulator-chain ILP**: codegen keeps **16** vector accumulators register-resident across the whole K-tile (`mr=8 × nr=16`, vl=8, all in `affine.for` iter_args, 0 in-loop stores); o3host uses **4** (`memref<4×vector<16xf32>>`). Zen4 has 2 FMA ports × ~4-cyc latency ⇒ needs ≥8 independent chains to saturate. codegen ≈100% FMA peak, o3host ≈50% (latency-bound). |
| QKV/out-proj | 32% | **codegen** | Same ILP (8-row vs 4-row), **plus** o3host does a BLIS A/B **operand-pack** copy pass that doesn't amortize at M=128 (panels already fit L2). codegen reads the weight in place, no pack. |
| QK^T, scores·V | 2.7% | **codegen** (negligible) | o3host **tiles the small-K loop** and round-trips the accumulator through the *heap* output buffer every K-tile (e.g. scores·V K=128 step 8 ⇒ 16 spills/tile, 4 needed); codegen keeps the K reduction in registers, writes once. Tiny FLOP ⇒ ~0 ms. |
| softmax exp | — | tie | Both scalar `math.exp`, 2.36M dynamic calls each (~8–16 ms), identical. The 864-vs-96 `expf` *call-sites* in codegen is benign static unrolling (codegen's fastmath let clang vectorize the surrounding reduction, scalarizing exp per lane) — **same dynamic work, not a regression**. |
| softmax/LN divide + scale | — | codegen (minor) | codegen stamps `fastmath<fast>` on its arith ops ⇒ clang lowers `x/y`→reciprocal-mul (**0 vdivps**) and `1/sqrt`→`vrsqrt14`+NR (**0 vsqrtps**); o3host has no fastmath ⇒ true `vdivps` (672) / `vsqrtps` (192), ~5× slower throughput. ~3–8 ms model-wide. |
| Gelu (powf+tanh) | — | **tie** | Both lower `powf(x,3.0)` and `tanh` to **scalar libm PLT calls**, one per element (288 each in both `.s`). Identical cost ⇒ 0 contribution to the gap — but huge *shared* absolute cost (§5). |

**Where the 85.6 ms (codegen vs o3host) comes from:** the GEMMs (≈70–80 ms,
chain-count ILP + no-pack), then the fastmath reciprocal/rsqrt eltwise (≈3–8
ms). Attention and transcendentals contribute ≈0 to the *gap* (tie/negligible).

**codegen's own headroom on the GEMMs** (does not change who wins, raises the
ceiling): it emits **vl=8/ymm only** (0 zmm) while o3host uses some vl=16/zmm —
codegen leaves a ~1.1–1.2× *instruction-count* reduction on the table (Zen4 zmm
has equal FP throughput but half the µops). And it K-tiles at `step 256`,
re-streaming the C accumulator tile 3–12× per (M,N) tile; the C tiles fit L2, so
a single large `Kc` would cut C round-trips to 1.

---

## 4. Backend gap (o3host vs o3emitobj): 1.66×, and it is NOT clang -O2/-O3

Same onnx-mlir --O3 krnl IR, two backends:
- **o3host** (0.372 s): `--EmitMLIR` → `convert-krnl-to-affine` → host
  `lower-affine` → `convert-krnl-to-llvm` → `mlir-translate` → `clang`.
- **o3emitobj** (0.224 s): onnx-mlir's native `--EmitObj` (krnl→llvm in-process).

**1.66× faster through onnx-mlir's own backend on byte-identical-transform IR.**
Bumping the host backend to `clang -O3` closes almost none of it (o3host
0.372→0.367, codegen 0.286→0.274) — so it is **not** the LLVM opt level. The
penalty is the `krnl→affine→scf→llvm` round-trip our pipeline forces (matches
the known finding that `convert-krnl-to-affine` degrades onnx-mlir --O3 ~1.85×).

**This penalty hits codegen too** — codegen and o3host both pay it, which is why
the §3 transform comparison is fair, but also why codegen's absolute 0.286 s
sits ~1.6× above where the same transforms would land through a native krnl→llvm
backend (≈0.18 s, which would *beat* o3emitobj's 0.224 s). Suspected causes
(unconfirmed): lost alignment/`noalias`/TBAA metadata across the textual MLIR
hand-off, or weaker affine→scf→llvm loop/vector lowering than onnx-mlir's
krnl→llvm. **This is the single largest lever (1.66× on the whole model) but the
hardest.**

---

## 5. Shared headroom: scalar transcendentals (helps us, not the gap)

Identical in all configs, so they don't move the codegen-vs-o3host gap — but
they are a large slice of *absolute* runtime that we can attack because we own
the MLIR:

| op | site | dynamic / inference | est. scalar cost | note |
|---|---|---|---|---|
| `powf(x, 3.0)` | Gelu x³ | 4.72M | ~70–180 ms | **`cst_2 = 3.0`** — should be `x*x*x`; *no config strength-reduces it* (288 libm calls in every `.s`). The `math.powf` carries no fastmath flag, so clang won't lower it. |
| `tanh` | Gelu | 4.72M | ~57–140 ms | scalar libm; serializes the loop ⇒ surrounding arith stays scalar too. |
| `exp` | softmax | 2.36M | ~8–16 ms | scalar libm. |
| `sqrt` | LayerNorm | 3072 | <1 ms | negligible (codegen folds to rsqrt). |

Gelu (powf+tanh) ≈ **20–45% of codegen's 286 ms**, paid in full by both
compilers. `powf(x,3)→x*x*x` is a trivial, exact rewrite; vectorizing
exp/tanh needs a polynomial approximation (correctness-gated).

---

## 6. Ranked fix backlog (spike-gated; one local commit per step; never push)

Ordered by **leverage × tractability**. Gate: each must move
`onnx-run-bench.sh --configs none,codegen` median with norm-rel-err ≤ 1e-4.

1. **Strength-reduce `math.powf(x, k)` for small integer k → multiplies.**
   ✅ **LANDED (2026-06-17, commit `0c64eff`).** New pass `dr-math-strength-reduce`
   (`powf(x,3.0)`→`x*x*x` by exponentiation-by-squaring, exact). Wired into the
   codegen pipeline. **Measured A/B (back-to-back): codegen 278.6 → 263.7 ms
   (1.057×, −14.9 ms), norm-rel-err 6.8e-6, 12 `math.powf` → 0.** vs o3host
   1.30→**1.41×**; net vs o3emitobj 1.28→**1.18×**. Lit 224/0, default
   byte-identical. Smaller than the ~tens-of-ms estimate (glibc `powf(x,3)` has a
   fast path + the Gelu loop is partly BW-bound + `tanh` still dominates the loop)
   — but free, safe, and it unblocks #2 (the `x*x*x` muls vectorize once `tanh`
   does).

2. **Vectorize the eltwise transcendentals (Gelu/softmax).**
   ✅ **LANDED (2026-06-17, commit `aa15241`).** Added a `poly-approx` option to
   `dr-math-strength-reduce` that lowers `math.exp`/`tanh`/… to upstream
   vectorizable polynomial approximations (`populateMathPolynomialApproximation
   Patterns`) — no libm call barrier, so the GELU/softmax loops auto-vectorize.
   **A/B: codegen 260.2 → 239.2 ms (1.088×), norm-rel-err 5.9e-6, 0 libm calls
   in the binary.** Cumulative with #1: **286 → 239 ms (1.20×)**; vs o3host
   1.30→**1.56×**; net vs o3emitobj 1.28×→**1.07× behind** (~80% of the net gap
   closed). Turned out cheaper than a hand-written polynomial — the upstream
   patterns + clang auto-vec suffice. Default OFF (byte-identical). Lit 225/0.

3. **The 1.66× backend/lowering penalty (§4) — DIAGNOSED; it *was* #2.**
   The cheap probe settled it: the gap is **not** alignment/opt-level — it is the
   **scalar transcendentals**. onnx-mlir EmitObj emits **0** `tanhf`/`expf`/`powf`
   (native vector polynomials); our `krnl→affine→llvm→clang` path left 288/96/288
   libm calls. A backend flag is insufficient (`-fveclib=libmvec` vectorizes exp
   but **not** tanh — libmvec has no vector tanh; gave ~0 e2e gain). The MLIR-level
   fix (#2) is what closed it. **Residual after #2: codegen 239 vs o3emitobj 224
   (1.07×)** — the small remainder is likely the poly loops not *fully*
   auto-vectorizing at clang -O2 (try -O3 / register-block the eltwise) + minor
   GEMM µkernel differences (#4/#5). No separate krnl→llvm work needed.

4. **codegen GEMM: emit vl=16/zmm.** ❌ **SPIKED — NO-GO (2026-06-17).** vl=16
   gives 234 ms vs vl=8 231 ms (slower); zmm 52K→58K, vfmaddps 3763→4723 but no
   speedup — confirms the G1 "ymm==zmm on Zen4" result *also* at the instruction
   level (the wider VL adds peel/tail overhead with no throughput gain because
   Zen4 double-pumps zmm over 2×256 pipes). Keep vl=8.

5. **codegen GEMM: single large `Kc`.** ❌ **SPIKED — NO-GO (2026-06-17).**
   Big-L2 JSON (force `gemmBlocking` to stop K-tiling) gives 243 ms vs 231 ms
   (slower) — without K-tiling the 9 MB FFN B-panel streams from DRAM every
   i-block (bandwidth-bound). The model's Kc=256 is correct. Keep model tiling.

**Backend `-O3` (not in original backlog): ✅ ADOPTED.** clang `-O2`→`-O3` on the
poly IR: 239→231 ms (~3%), correctness unchanged (the now-call-free eltwise loops
let -O3's vectorizer do more). `-ffast-math` adds nothing (ops already carry
`fastmath<fast>`). Harness backend bumped to -O3.

6. **Fully vectorize the residual eltwise (Gelu poly arith).** ❌ **BUILT, then
   REVERTED — measured NO-GO + adversarially proven unsafe (2026-06-17).**
   - First, `affine-super-vectorize` *does* vectorize the poly eltwise but emits
     `vector.transfer_read` (no `in_bounds`) that onnx-mlir's krnl backend can't
     lower (forcing it → masked loads). So I built a dedicated
     `dr-eltwise-vectorize` pass emitting clean `affine.vector_load`/`store` (the
     register-block convention), guarded for innermost/pointwise/divisible/
     stride-1/no-iter_args/same-address loops. Correct on the openai-gpt e2e
     (norm-rel-err 5.8e-6), lit-tested.
   - **Performance: SLOWER.** Back-to-back A/B: base (clang -O3 auto-vec) ~233 ms
     vs +eltwise-vec ~239 ms (~3% slower). clang -O3 already auto-vectorizes the
     call-free poly loops, likely with better width/unroll/scheduling than a fixed
     vl=8 explicit form; pre-vectorizing constrains it and adds overhead. So the
     residual scalar `*ss` ops were *not* the bottleneck — we were already at
     parity via clang's auto-vec.
   - **Correctness: a 12-agent adversarial review found 5/6 lenses with confirmed
     critical miscompiles** (runtime-reproduced): `iv` in a non-innermost map
     result (`x[k,k]` diagonal → wrong contiguous load); cross-lane recurrence via
     memrefs that *alias* through distinct SSA values / function args (the
     same-address guard keys on SSA value, not buffer — needs real alias
     analysis); iv-as-value and vector-of-non-scalar-type holes. A *correct*
     general eltwise vectorizer is substantially harder than it looks.
   - **Verdict: reverted.** A pass that doesn't beat clang -O3 *and* needs real
     alias analysis to be safe is not worth shipping. clang -O3 auto-vec is the
     right tool for the eltwise here; we are at parity with o3emitobj without it.

6. **Fold zero-init passes into the first K-iteration** (QK^T/scores·V write a
   full zero pass over the score/out buffer before accumulating — the same
   init-fusion `canonicalizeAllocaGemm` already does for the big GEMMs). Tiny
   (attention is 2.7% FLOP); do for completeness only.

**Out of scope** (confirmed not levers here): clang `-O2`→`-O3` (§4); attention
matmul kernels (2.7% FLOP, already won); batch>1; threading (both single-thread).

---

## 7. Reproduce

```bash
cd drcc-benchmarks/onnx/scripts
ONNX_MLIR_IMAGE=onnx-mlir-lean ONNX_MLIR_TAG=x86_64 \
DRCC_IMAGE=drcc-lean DRCC_TAG=x86_64 \
DR_OPT_HOST=/…/onnx-mlir/build/tools/dr-opt/dr-opt \
./onnx-run-bench.sh /…/onnx/models/openaigpt_Opset18.onnx \
  --configs none,codegen,o3host \
  --cost-model /…/onnx-mlir/bench/zen4-gemm.json \
  --iters 15 --warmup 3 --keep-ir --out-dir /tmp/cg-ogpt -v
```

o3emitobj (onnx-mlir's own backend) and the clang-O3 backend probe:
`/tmp/cg-ogpt/measure-o3emitobj.sh`, `/tmp/cg-ogpt/test-o3backend.sh`.
Census/disasm builder: `/tmp/cg-ogpt/build-specimens.sh`.
