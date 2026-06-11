# Cost-Model Tiling Investigation — Findings

**Date:** 2026-06-03
**Author:** Claude (Opus 4.8)
**Scope:** Whether the drcompiler unified cache+register cost model, driving affine loop tiling, yields a defensible CGO result on real PolyBench — and where/why it does not.

**Method note:** All real-kernel numbers below use **real cgeist-lowered PolyBench** (not hand-written MLIR), compiled through the `drcc` Docker image's cgeist (LLVM 18) → host `dr-opt` (the fixed cost model) → host `mlir-opt` lowering → `clang -O3 -march=native -ffast-math`. Measured on AMD Ryzen 9 7950X3D, core-pinned (`taskset -c 8`, the 32 MB-L3 CCD), median of 3, LARGE dataset unless noted.

---

## Executive summary

After fixing real bugs in the tiling cost model and testing it rigorously on real PolyBench, **the cost-model affine tiling does not yield a "we beat X" result against any defensible baseline** — not Polly, clang, gcc, `-O2`, or even upstream MLIR's own placeholder tiling. The investigation produced three genuinely useful things, none of which is a headline on its own:

1. **Real correctness/perf bug fixes** in the tiling cost model (kept; tests green).
2. **One verified data point** — seidel-2d, where the model makes the best MLIR tiling decision (2.5× over no-tile, 1.3× over the placeholder, output verified correct).
3. **A rigorous negative/characterization result**: *why* analytical affine tiling caps below production compilers, and why per-kernel tile selection resists a simple cost model.

---

## 1. The hand-written artifact (the first lesson)

An initial spike used **hand-written** affine GEMM MLIR and concluded the fixed cost model "beats Polly by 26%." This was an **artifact**: the hand-written no-tile MLIR compiled to pathologically slow scalar code, making any tiling look heroic. On **real cgeist** PolyBench the effect vanished. *Lesson: benchmark transformations on real frontend-lowered code, never hand-written IR.*

---

## 2. Real PolyBench results (fair, `-ffast-math` everywhere)

| LARGE, core8 | cost-model | no-tile | clang-O3 | gcc-O3 | Polly |
|---|---|---|---|---|---|
| gemm | 0.301 | 0.301 | **0.087** | 0.101 | **0.066** |
| syrk | **0.176** | 0.175 | 0.308 | 0.300 | 0.555 |
| 2mm | 1.66 | 2.35 | 0.725 | 0.961 | **0.064** |
| jacobi-2d | 0.688 | 0.649 | 0.619 | 0.650 | 1.072 |
| heat-3d | 0.892 | 0.854 | 0.853 | **0.716** | 0.919 |
| seidel-2d | **4.80** | 12.1 | 3.62 | 3.62 | absent |

- We **beat Polly** on syrk (3×, incidental — Polly over-transforms it) and seidel (Polly can't tile the stencil).
- We **lose** on gemm and 2mm — the kernels that matter — by 3.5–25×.
- The **tiling itself** only helps on 2 of 6 (seidel 2.5×, 2mm 30%); it is a no-op on gemm/syrk/jacobi and *hurts* heat-3d.

### `-O2` does not change this
clang's loop vectorizer + unroll-and-jam run at `-O2` too, so **clang -O2 (0.087) ≈ clang -O3 (0.088)** on gemm. The only baseline `-O2` slows is gcc; clang -O2 and Polly -O2 still beat us 3.5×. The "immature backend" framing fails because the cost-model path *uses clang as its backend* — the gap is the lowering, not the backend.

---

## 3. The gemm cap: scalar lowering, not tiling

On gemm the MLIR path is 3.5× slower than clang **at every tile size 8→256** (all ≈0.300 s). Diagnosis:

- `perf`: MLIR path = **2,745 M instructions**; clang = **590 M** (4.6× more), only 331 K L1 misses → **instruction-bound, not memory-bound**.
- The matmul loop **is vectorized** (width-8 zmm) but has a **1:1 load-to-FMA ratio** — **no register blocking**. clang/Polly keep an m×n block of `C` accumulators in registers and reuse each `A`/`B` load; the MLIR-lowered loop reloads operands every FMA.
- **Standard MLIR passes do not fix it:** `affine-scalrep` (no-op), `affine-super-vectorize` (emits 0 vector ops), `affine-loop-unroll-jam` ×4/×8 (no change), `affine-loop-unroll` ×8 (2.7× *worse*).

**Conclusion:** the MLIR affine→LLVM lowering does not generate a register-blocked micro-kernel, and the off-the-shelf passes don't add one. Closing this is a multi-week GEMM-codegen project (re-implementing what BLIS/Polly/clang do), independent of the cost model. It caps the whole approach below the production compiler on dense LA regardless of tile choice.

---

## 4. MLIR-vs-MLIR: we don't beat the placeholder either

The fairest test — same lowering + backend, our tiling vs upstream MLIR's nth-root placeholder vs no-tile:

| LARGE | no-tile | upstream-tile | cost-model | vs upstream |
|---|---|---|---|---|
| gemm | 0.300 | 0.299 | 0.299 | tie |
| syrk | 0.175 | 0.178 | 0.176 | tie |
| 2mm | 2.33 | 1.652 | 1.655 | tie |
| jacobi-2d | 0.649 | **0.618** | 0.688 | **lose** (we regressed below no-tile) |
| seidel-2d | 12.1 | 6.38 | **4.80** | **win 1.3×** |
| heat-3d | 0.919 | **0.870** | 0.892 | lose |

**Scorecard vs the placeholder: 1 win, 2 losses, 3 ties.** On jacobi the cost model made a *harmful* decision (slower than no tiling) where the one-line heuristic correctly helped.

### Why we lose: the register term is matmul-specific
Our model picks **tile≈8 uniformly** because the register tension assumes a **c² accumulator block** (`c²≤regCap=128 ⇒ c≤8`). That is correct for matmul but wrong for stencils, which keep only a constant input window live. The forced-tile sweep proves it: jacobi's true optimum is **tile≈100** (a large cache tile, 0.560 s), which our model rejects as "too much register pressure" and clamps to 8. Upstream's footprint-only nth-root (~25) ignores registers and lands closer.

### The conditional fix backfired
Charging the c² penalty only for detected reductions (freeing stencils to pick large cache tiles) **fixed jacobi but destroyed seidel** (4.80 → 8.80): seidel is a stencil that *wants* the small tile. seidel and heat want small; jacobi wants large — all stencils. **"Stencil vs reduction" does not predict the right tile.** No one-line classifier separates them, because the optimum depends on per-kernel reuse/dependence/cache-conflict behavior the analytical model does not represent. Reverted.

---

## 5. What is verified true

- **seidel-2d** cost-model tiling: checksum-verified correct (matches clang -O0 ground truth to 10 digits), 2.5× over no-tile, best MLIR decision. A real but single data point.
- The **bug fixes** (32-bit overflow → double; tile-invariant-ALU domination removed; degenerate tile-2 → tile-8 register tension; uniform emission) are real and correct; lit tests updated and green.

---

## 6. The fusion fork hits the same wall

The project's *original* premise was replacing MLIR's **fusion** cost model — the one carrying the literal `// TODO: This is a placeholder cost model`. Fusion is a different mechanism (eliminate intermediate-array materialization), so it was tested separately as a fresh angle. Same lowering + backend; cost-model fusion vs upstream placeholder fusion vs no-fusion:

| LARGE, core8 | no-fusion | upstream-fusion | cost-model-fusion | clang | polly |
|---|---|---|---|---|---|
| 2mm | 2.320 | 2.330 | 2.320 | 0.725 | 0.064 |
| atax | 0.00468 | 0.00466 | 0.00466 | 0.00135 | 0.00301 |
| gemver | 0.00897 | 0.00823 | 0.00821 | 0.00503 | 0.00654 |
| bicg | 0.00090 | 0.00090 | 0.00090 | 0.0102 | 0.00280 |

- **cost-model fusion ≈ upstream placeholder fusion everywhere** — both fuse the same loops (2mm: 10→8 affine.for for both; output checksums verified identical), no improvement.
- **Fusion barely helps**: no-op on atax/bicg, 9% on gemver, and **zero on 2mm** — its materialization win is masked because 2mm is compute-bound under the scalar lowering (same cap as §3).

So **both forks — tiling and fusion — show the cost model tied with MLIR's built-in heuristics on real PolyBench.** The "replace the placeholder" thesis does not hold empirically for either transform.

## 7. Meta-conclusion

Every adjustment to the tile cost model was **whack-a-mole** — fixing one kernel broke another — because the model is a coarse proxy for effects (register pressure, cache conflicts, prefetching, the backend's own unroll/vectorize choices) that genuinely determine the optimum. That is not a bug; it is the reason analytical tile-size selection is hard and why polyhedral/autotuning systems exist.

The evidence across all framings — artifact, scalar-lowering cap on dense LA, `-O2` parity, 1-win/2-loss/3-tie vs the placeholder, the conditional-fix backfire — is unambiguous: **the cost-model affine-tiling direction does not contain a "we beat X" CGO result, and no incremental fix changes that.**

The honest, defensible outputs are the bug fixes, the verified seidel data point, and this characterization. A *positive* CGO claim would require either closing the scalar-lowering gap (a separate GEMM-codegen project) or moving the contribution off "beat the optimizer at tile selection" entirely.
