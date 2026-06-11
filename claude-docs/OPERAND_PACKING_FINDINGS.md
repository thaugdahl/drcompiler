# Operand Packing Spike — and the bigger thing it uncovered

**Date:** 2026-06-05
**Author:** Claude (Opus 4.8)

**TL;DR:** Operand packing is **not** the lever (it's irrelevant to both shortfalls).
But chasing "are there no stones unturned?" overturned the earlier **rank-k loses**
verdict: register-blocking **does** generalize to rank-k (syrk **2.49×**, syr2k
**1.63×** over clang -ffast-math) — the earlier loss was a **wrong-configuration
artifact**, not an algorithm limit. The real finding is a **configuration space**:
register-blocking's win requires selecting `(orientation, tile, reassoc)` from the
kernel's operand-stride layout, and the two BLAS-3 families sit at **opposite
corners**. No packing anywhere.

All numbers: Ryzen 7950X3D core 8 (32 MB L3), N=1024, checksum-verified, median of
5, clang baseline `-O3 -march=native -ffast-math` (its real best loop order).

---

## Part 0 — packing is a red herring (both shortfalls)

- **Rank-k:** packing `A[j][k]→Apack[k][j]` (stride-1 in j) + the broadcast micro-
  kernel reached only **0.70× clang** (15.4 vs 22.0). The pack was 6% of time; the
  micro-kernel itself was the ceiling. Packing fixed the *stride* but not the
  problem (the problem was the *orientation* — see Part 1).
- **Large-N gemm:** BLIS ladder past LLC (N=2048/4096): cache-tiling is the lever
  (**1.76× / 2.48×** over clang), packing-on-top adds only **+2.5–3%**. And the
  pass already cache-tiles. Packing's marginal value over what the pass does ≈ 3%.

**Do not build operand packing into the pass.**

## Part 1 — the real result: a (orientation, tile, reassoc) configuration space

Register-blocking holds an mr×nr block of C in registers across the k-reduction.
*Which* loop LLVM ends up vectorizing — and therefore whether the kernel wins —
depends on three coupled choices the pass must make from operand layout:

| config axis | `C=A·B` (gemm/2mm/3mm) | `C=A·Aᵀ` (syrk/syr2k) |
|---|---|---|
| 2nd operand stride-1 in | spatial **j** (`B[k][j]`) | reduction **k** (`A[j][k]`) |
| LLVM strategy that wins | **SLP over independent j-lanes** | **reduction vectorization over k** |
| reassoc (fast-math) needed | **no** (lanes independent) | **yes** (reorders the k-sum) |
| best tile | **wide 8×16** | **small square** (syrk 4×4, syr2k 2×2) |

The two families are at **opposite corners**: a single fixed configuration cannot
serve both. This is *the* contribution refinement — layout-aware register-blocking.

### Measured (best config per kernel, all checksums match)

| kernel | family | best config | rb GFLOPs | clang | **speedup** |
|---|---|---|---|---|---|
| gemm  | C=A·B  | 8×16, no-reassoc | 47.97 | 21.19 | **2.26×** |
| 2mm   | C=A·B  | 8×16, no-reassoc | 48.25 | 20.31 | **2.38×** |
| 3mm   | C=A·B  | 8×16, no-reassoc | 49.69 | 20.38 | **2.44×** |
| syrk  | C=A·Aᵀ | 4×4, **reassoc**  | 55.37 | 22.23 | **2.49×** |
| syr2k | C=A·Aᵀ | 2×2, **reassoc**  | 26.72 | 16.37 | **1.63×** |

**Register-blocking generalizes across all tested BLAS-3, 1.6–2.5×.** rank-k was
never an algorithmic loss.

### Why the earlier "rank-k loses 3–5×" was wrong (two artifacts)

1. **Tile.** The gemm-tuned **8×16 = 128 accumulators** is correct for the
   broadcast form (16 zmm: nr=16 → 2 zmm/row × 8 rows). In the dot form each
   accumulator is a *full* zmm, so 8×16 → 128 zmm → catastrophic spill (3.0 GFLOPs).
   The dot form needs a small square grid (≤ 32 zmm on Zen4): syrk 4×4 (16 acc + 8
   loads = 24 zmm), syr2k 2×2 (4 input streams/(i,j) → tightest).
2. **reassoc.** MLIR lowering emits **flagless** FP ops; `clang -ffast-math` does
   **not** retroactively flag ops in a `.ll`. LLVM's loop vectorizer refuses to
   reorder a FP reduction without `reassoc` → the k-reduction stays **scalar**
   (syrk pass-output: 42 vector vs **258 scalar** FMAs → 14.4 GFLOPs). Adding
   `fast` to the IR FP ops: **452 vector** FMAs → **53.97 GFLOPs**. The MLIR path
   was being *handicapped* vs the clang -ffast-math baseline, not the reverse.

### Mechanism, disassembly-confirmed

The same gemm pass output, two flag regimes:
- **no-reassoc:** 48 vector pd, **0 shuffles** → SLP packs independent C[i][j] lanes,
  each accumulated over k with no cross-lane work. → **47.97**.
- **+reassoc:** 408 vector pd, 128 scalar, **384 shuffle/perm** → LLVM switches to
  *loop-vectorizing the k-reduction* (horizontal sums = the 384 shuffles), which is
  right for rank-k but **catastrophic** for the broadcast layout. → 16.52.

So reassoc *flips* the vectorizer strategy. rank-k **needs** the flip; `C=A·B` is
**ruined** by it. Hence the pass must pick reassoc per layout, not globally.

## Implications for the pass (the build work)

The transform itself is already correct for both families (unroll-jam the two
spatial loops, promote the C accumulators to iter_args over k). What's missing is
**layout-driven configuration selection**:
1. **Detect the family** from the reduction body: is the non-`A[i][k]` multiplicand
   stride-1 in the spatial unroll dim (j → `C=A·B`) or in the reduction dim (k →
   rank-k)?
2. **Pick the tile** accordingly: wide (8×16) for broadcast, small square (≤4×4)
   for dot; shrink further as the input-stream count rises (syr2k → 2×2).
3. **Set `fastmath<fast>`** (reassoc) on the register-blocked FP ops **iff** dot
   orientation; leave `C=A·B` un-reassoc'd.

(2)+(3) are small; (1) is the real new logic. No packing, no new copy nests.

## Recommended next step

This is now a strong, defensible **"register-blocking generalizes across BLAS-3 via
layout-aware configuration"** story. Two paths: (a) implement the selection above so
the pass auto-wins all five; (b) proceed to the SOTA study (Bondhugula MLIR-GEMM /
Polly / BLIS) on the established wins. (a) is the higher-value next step — it closes
the rank-k gap *in the pass*, not just in hand-spikes.

## Reproduce
- Rank-k orientation discovery: `/tmp/claude/gen/syrk_dotrb_k.c` (hand-C dot block,
  57.5), `syrk_dotrb_param.c` (tile sweep). Pass path + fast-math: `run_ff.sh`
  (8×16) and `run_ff44.sh` (4×4); non-fast baseline `run.sh`. syr2k tile sweep
  inline. Fast-math injected by sed on the mlir-translate `.ll`
  (`s/= fadd double/= fadd fast double/` etc.) — proxy for emitting `fastmath<fast>`.
