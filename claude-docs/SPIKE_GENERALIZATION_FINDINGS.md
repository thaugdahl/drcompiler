# Generalization Spike — does register-blocking generalize across BLAS-3?

**Date:** 2026-06-05
**Author:** Claude (Opus 4.8)
**Question (user):** Does the GEMM 2.4× generalize across the reduction class?
Then: **broaden within BLAS-3** — make the triangular rank-k kernels fire too.

**Verdict (SUPERSEDED 2026-06-05 PM — see OPERAND_PACKING_FINDINGS.md):** The
"rank-k loses 3–5×, it's a `C=A·B`-only technique" conclusion below was **WRONG**.
It was a wrong-*configuration* artifact, not an algorithm limit. With the correct
configuration (dot orientation, small square tile, reassoc/fast-math), **rank-k
WINS too**: syrk **2.49×**, syr2k **1.63×** over clang -ffast-math, checksum-
verified. Register-blocking generalizes across **all** tested BLAS-3 (1.6–2.5×);
operand packing is irrelevant. The real result is a `(orientation, tile, reassoc)`
**configuration space** selected from operand layout — the two families sit at
opposite corners. **Read OPERAND_PACKING_FINDINGS.md for the corrected picture.**
The triangular diagonal-peeling engineering below is still valid and used.

---

**Original (now-superseded) verdict:** Register-blocking on the memref/affine path is a
**`C = A·B` (matmul-multiplication) technique**. It generalizes cleanly across
the matmul family — **gemm, 2mm, 3mm: ~2.2–2.35× over clang's best loop order**,
checksum-verified. It does **not** profit the **rank-k family (syrk, syr2k)**:
the pass now *fires* on them (triangular diagonal-peeling implemented + correct),
but register-blocking **loses to clang 3–5×** there, for a principled reason
(below). Honest, useful boundary — not "works on everything."

---

## Result (N=1024, Ryzen 7950X3D core 8, opaque standalone kernels, checksum-verified)

Same cgeist front-end + same MLIR lowering + same `clang -O3 -march=native
-ffast-math` backend for all three configs; they differ only in the dr-opt pass.
**Each kernel is written in clang's best loop order** (matmul family i-k-j /
stride-1 inner j; rank-k i-j-k / k-innermost dot product) so the clang baseline
is its real ceiling — no strawman. GFLOPs, higher better.

| kernel | family | fired | correct | naive-MLIR | **rb** | clang-O3 (best order) | rb/clang |
|---|---|---|---|---|---|---|---|
| gemm  | C=A·B  | YES (1) | ✓ | 20.3 | **46.4** | 20.7 | **2.24×** |
| 2mm   | C=A·B  | YES (2) | ✓ | 20.1 | **43.2** | 20.2 | **2.14×** |
| 3mm   | C=A·B  | YES (3) | ✓ | 20.0 | **46.7** | 20.1 | **2.32×** |
| syrk  | C=A·Aᵀ | YES (1) | ✓ | 3.5  | 7.3  | **22.3** | **0.33×** |
| syr2k | C=A·Aᵀ | YES (1) | ✓ | 6.7  | 3.5  | **17.6** | **0.20×** |

- **Matmul family (gemm/2mm/3mm):** register-blocking fires (one `iter_args`
  micro-kernel per matmul band; 2mm/3mm chain — an intermediate is produced then
  consumed) and wins **~2.3× over clang's best**, checksums identical. The 2.4×
  is not a GEMM artifact.
- **Rank-k family (syrk/syr2k):** the pass *fires* (diagonal-peeling makes the
  triangular nest register-block, checksums match) but **loses 3–5× to clang**;
  for syr2k it's even slower than the un-blocked MLIR (3.5 < 6.7) — register-
  blocking actively hurts.

## Why register-blocking helps `C=A·B` but not `C=A·Aᵀ`

Register-blocking holds an mr×nr block of C in registers and **vectorizes the
inner spatial loop `j`** (the nr dimension), broadcasting `A[i][k]` and streaming
the second operand across `j`.

- **`C=A·B` (gemm):** the second operand is `B[k][j]` — **stride-1 in `j`**, the
  vectorized dimension. clang already vectorizes inner-j but keeps C in memory
  (1 load+store per FMA → memory-bound, ~20 GFLOPs); register-blocking holds C in
  registers → compute-bound → **2.3×**.
- **`C=A·Aᵀ` (syrk):** the second operand is `A[j][k]` — **stride-1 in `k`, NOT
  `j`**. clang's best is the *other* formulation: k-innermost dot product
  (`A[i][:]·A[j][:]`), which it vectorizes over `k` to **~20 GFLOPs**. Register-
  blocking the k-innermost form unrolls `j` into scalar/SLP FMAs (the transposed
  operand can't be packed into stride-1 j-vectors without an explicit transpose),
  so it can't match clang's clean k-SIMD reduction.

**Conclusion:** profiting from register-blocking on rank-k updates requires
**operand packing** (transpose A into stride-1 panels — the BLIS approach), which
this pass does not do. Without packing, register-blocking is the wrong transform
for `C=A·Aᵀ`. This is the same "true BLIS ceiling" (packing/prefetch) noted as
unmeasured elsewhere.

## What was built this spike — triangular diagonal-peeling

`peelTriangularNest` (lib/Transforms/AffineRegisterBlock.cpp): given a perfect
band whose inner spatial bound depends on the outer IV (`for i { for j=0..i+1 {
red }}`), rewrite to a strip-mined rectangular HEAD + a scalar DIAGONAL:
```
for ii = 0 to N step mr {
  for i' = 0 to mr { for j = 0 to ii   { red(ii+i', j) } }   // HEAD: register-blocked
  for i' = 0 to mr { for j = ii to f(.) { red(ii+i', j) } }   // DIAG: scalar
}
```
The HEAD's inner bound `ii` is invariant in `i'`, so its unroll-and-jam is legal;
the DIAG keeps the ragged bound and is left scalar (Stage 3 skips it).

> **Two bugs caught + fixed during bring-up (checksum/crash, not "valid IR"):**
> 1. **Dangling `sOut`.** `loopUnrollJamByFactor` ends with
>    `promoteIfSingleIteration`. The peeled head-i' has trip == mr, so unroll-
>    jamming it by mr makes it single-iteration → the loop is **promoted away and
>    erased**, leaving the old `sOut` handle dangling; the subsequent
>    `findReductionLoopUnder(sOut)` walked freed memory → segfault. gemm's `i`
>    (trip ≫ mr) survives, which is why it never showed. Fix: re-find the
>    reduction by walking the stable `func` (findReductionLoopUnder skips
>    already-promoted reductions, so it stays correct across 2mm/3mm).
> 2. **Apply between jammed loops.** unroll-jam can't thread a non-loop op (the
>    `ni = ii + i'` row offset) sitting *between* the loop being jammed and the
>    jammed inner loop → put the offset INSIDE the j-loop so head-i' is a perfect
>    single-child nest.

## Methodology confounds caught (the honest-verdict discipline)

1. **Init forward-propagation:** compiling the *whole* PolyBench program through
   cgeist -O2/-O1 forwards `init_array`'s closed form into the kernel — the
   "matmul" recomputes A/B instead of loading them. Switched to standalone opaque
   kernels (static `1024x1024` via a `memref<?x→1024x` sed).
2. **Loop-order strawman (caught TWICE):** first wrote 2mm/3mm i-j-k (strided)
   → clang 1.1 GFLOPs, rb looked 42–85×. Then syrk/syr2k i-k-j → clang 0.59, rb
   looked 12×/6.7×. Both inflated by the *interchange* the pass does and clang
   doesn't. Rewriting each kernel in clang's best order gave the honest numbers
   (2.3× for matmul; **clang wins** for rank-k).

## CGO implication

The breadth push produced a **sharper, more defensible boundary** than "more
kernels": register-blocking on the un-raisable memref/affine path is a
`C=A·B` technique (gemm/2mm/3mm, ~2.3× over clang), and rank-k updates need
operand packing it doesn't do. A reviewer's "does it generalize?" is answered
precisely — *yes within the matmul family, and here's the exact reason it stops*.
Natural next step toward the BLIS ceiling: operand packing (would let rank-k and
large-N both win, and is the honest gap vs hand-tuned BLAS).

## Validation
- Lit tests (all PASS): gemm-register-block, gemm-ikj-imperfect, gemm-cache-tile,
  gemm-imperfect-cache-tile, **syrk-triangular-peel** (new — proves the triangular
  nest register-blocks: strip + 2×2 `iter_args` head + scalar diagonal).
- Spike harness: `bash /tmp/claude/gen/run.sh` (`*_k.c`/`*_m.c` in `/tmp/claude/gen/`,
  gemm reuses `/tmp/claude/pb_gemm.c`+`pb_main.c`).
- Pass: `affine-register-block{mr=8 nr=16 cache-tile=true mc=256 nc=256 kc=256}`.
