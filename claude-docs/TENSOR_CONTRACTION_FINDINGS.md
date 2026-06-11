# Tensor contraction spike — does register-blocking generalize past 2D matmul?

**Date:** 2026-06-05
**Author:** Claude (Opus 4.8)
**Ask (user):** Push register-blocking toward tensor contractions (batched matmul,
tensor-times-matrix, multi-reduction / conv-shaped, einsum).

**Result (one line):** Register-blocking generalizes to **tensor contractions**
broadly — 2D-output (incl. multi-reduction) **and batched matmul** — at
**2.7–3.4×**, once the broadcast micro-kernel is emitted via the **vector
dialect** (mr rows × `vector<vl>` accumulators) instead of relying on LLVM SLP.
The earlier ≥3D-output "wall" was an SLP-on->2D-addressing limit; explicit
vectorization removes it.

**Update (2026-06-05, later):** the vector micro-kernel is now built into the
pass (default for the broadcast family). Batched matmul went from **0.66× (loss)
→ 2.68×**, ttm **0.68× → 3.36×**, multi-reduction tcon2 **2.62× → 3.21×**, all
checksum-verified; BLAS-9 unchanged; lit 8/8. See "The fix that landed" below.

All numbers: Ryzen 7950X3D core 8, vs `clang -O3 -march=native -ffast-math` best,
checksum-verified (`MATCH`). Tensor kernels cubic N=128 (memory); BLAS N=1024.

---

## Measured

| kernel | shape | output rank | fires | correct | rb/clang |
|---|---|---|---|---|---|
| gemm (ref)  | `C[i][j]=ΣA[i][k]B[k][j]`            | 2D | yes | MATCH | 2.36× |
| **tcon2**   | `D[a][c]=Σ_{k1,k2} X[a][k1][k2]·Y[k1][k2][c]` | **2D** | yes | MATCH | **2.62×** |
| covariance  | `cov[i][j]=Σ D[k][i]D[k][j]`          | 2D | yes | MATCH | 2.60× |
| bmm         | `C[b][i][j]=Σ A[b][i][k]B[b][k][j]`   | **3D** | yes | MATCH | 0.66× (cap ~1.2× w/ small tile) |
| ttm         | `D[a][b][c]=Σ X[a][b][k]Y[k][c]`      | **3D** | yes | MATCH | 0.68× (cap ~1.2×) |

## What generalizes — and the precise reason

The pass blocks **2 free (spatial) dims** as the mr×nr register tile and treats
everything else as reduction or outer loops. So generalization hinges on the
**output (accumulator) being expressible as a 2D block**:

- **Multi-reduction is free.** tcon2 has *two* reduction dims (k1,k2). The pass
  blocks the 2D output (a,c) as 8×16 and promotes the accumulator across the
  **innermost** reduction (k2); the outer reduction (k1) remains a loop that
  re-streams the block. Suboptimal in theory, **2.62×** in practice (LLVM
  vectorizes the 8×16 grid: 32 vector / 0 scalar FMAs). conv is this shape
  (reduction over c,kh,kw; 2D output over the spatial map) → should follow.
- **2D-output is the invariant**, not the operand rank: tcon2's *operands* are
  3D (`X[a][k1][k2]`, `Y[k1][k2][c]`) yet it wins, because its **output** `D[a][c]`
  is 2D. The accumulator block is what lives in registers.

## The boundary — ≥3D output (batched matmul), an SLP wall

bmm/ttm fire and are correct, but the default 8×16 tile goes **scalar** (15 vector
/ 98 scalar FMAs) → 0.66×. Controls pin the cause precisely:
- **Not small-N:** plain gemm at the same N=128 wins **2.68×** (48 vector FMAs).
- **Not operand rank:** tcon2 (3D operands, 2D output) wins.
- **It is the 3D *accumulator*:** every block load/store carries the extra batch
  index (`A[b][i][k]`, `C[b][i][j]`), and LLVM's SLP fails to prove the 16-wide
  innermost face contiguous through the >2D addressing → scalarizes.
- **Register pressure compounds it:** a smaller tile recovers *some* (sweep:
  8×16→0.66×, 4×8→1.0×, **2×16→1.1–1.2×**) but never reaches 2D's 2.3× — the
  big tile that wins in 2D spills/scalarizes with the extra address registers.

So batched matmul is a **codegen-quality wall (MLIR affine→LLVM + SLP), not a
transform limit.** Two routes out (the open research fork):
1. **Collapse the batch for collapsible shapes.** ttm `D[a][b][c]=X[a][b][k]Y[k][c]`
   → `memref.collapse_shape` (a,b)→ab gives `D[ab][c]=X[ab][k]Y[k][c]` = *exactly
   gemm* (2D output) → should hit 2.3×. Works when the batch isn't shared across
   all operands (ttm yes; true bmm no — b indexes all three).
2. **True batched (bmm):** needs the 2D-per-batch block to vectorize through the
   3D memref — e.g. precompute the per-batch base pointer / subview so the inner
   micro-kernel sees a 2D contiguous face, or emit the micro-kernel via the vector
   dialect rather than relying on LLVM SLP.

## Bug fixed this spike — degenerate cache-tiling at small N

`tilePerfectlyNested` with a tile larger than the loop extent emits `step >N`
loops with min/max (`#map`) point bounds that the LLVM vectorizer can't analyze →
the micro-kernel goes scalar (tcon2 collapsed to **0.15×** with the default
mc=nc=kc=256 at N=128). **Fix:** only tile a band when some dim genuinely exceeds
its tile (`extent > tile`); otherwise register-block untiled. After the fix tcon2
wins **2.62× at default flags**; BLAS N=1024 unchanged (gemm 2.36×, syrk 2.58×,
covar 2.60×); lit 7/7.

## The fix that landed — vector-dialect broadcast micro-kernel

Spike first (clang vector extensions, validates the approach): an explicit-vector
bmm micro-kernel hit **3.37×** and ttm **3.69×** (vs SLP's 0.66×/0.68×) — proving
the wall is purely SLP, not the transform. Then built into the pass:

For the broadcast family the pass now emits, instead of unroll-jam-by-nr +
scalar-promote + SLP:
- unroll-jam the outer spatial loop by `mr` (mr rows),
- vectorize the inner spatial loop to `vector<vl>` (vl=8 = one zmm): `mr` vector
  accumulators carried over the reduction via `affine.for` iter_args,
- the streamed operand → contiguous `affine.vector_load`, the broadcast operand →
  scalar load + `vector.broadcast`, each FMA → vector `arith.mulf`/`addf`,
- `affine.vector_store` the results.

Because vectorization is now *explicit*, it survives arbitrary tensor-contraction
addressing — the >2D accumulator no longer defeats it. Measured (from the pass,
checksum-verified):

| kernel | SLP (before) | vector µkernel (now) |
|---|---|---|
| bmm (batched matmul) | 0.66× | **2.68×** |
| ttm (tensor×matrix)  | 0.68× | **3.36×** |
| tcon2 (multi-reduction) | 2.62× | **3.21×** |

Falls back to scalar+SLP when a band isn't cleanly vectorizable (non-constant
bounds — e.g. cache-tiled point loops, gather access). So cache-tiled large-N
gemm stays on the SLP path (2D, works); the dot/rank-k family is untouched
(vectorizes over the reduction via fastmath). `vectorize` (default true) / `vl`
(default 8) options; `vectorize=false` reproduces the scalar+SLP path.

Pipeline note: the lowering must now include `--convert-vector-to-llvm` (the pass
emits vector dialect). Added to the bench harness; the production drcc pipeline
needs the same.

## Status / next
- **Shipped:** cache-tile degenerate guard; **vector-dialect broadcast
  micro-kernel** (default, rank-gated); large-N composition; **trmm
  reduction-peel**. Tensor contractions — 2D-output, multi-reduction, AND batched
  matmul — register-block at **2.7–3.4×**. BLAS-9 unchanged; lit 9/9.

### Follow-ups resolved (the "go all except SOTA" round)
- **(a) Cache-tiled point loops — DONE + rank-gated.** `getConstantTripCount`
  replaces the `hasConstantUpperBound` check, so the vector µkernel composes with
  cache tiling (tiled point loop `tc..tc+tile` has a constant trip; a partial last
  tile has a non-constant trip → safe SLP fallback). But measurement showed the
  vector µkernel **underperforms SLP for a 2D accumulator** (gemm N=1024 tiled:
  SLP 2.3× vs vector 1.7–2.15×; gemm N=2048: **SLP 5.2×** vs vector 3.7×). So the
  vector path is **rank-gated**: accumulator rank ≥3 → vector µkernel (the only
  case SLP fails); rank ≤2 → SLP (better at every size). Net: large-N 2D gemm
  **5.2×** via SLP+cache-tile, large-N batched matmul gets cache-tile + vector
  µkernel, **no 2D regression**. Lesson: SLP wins 2D; the explicit vector µkernel
  is needed *only* for the >2D accumulator.
- **(b) trmm triangular-reduction peel — DONE.** `peelTriangularReduction`: for
  `for i { for k=i..N { for j { red }}}` (reduction lower bound on the outer IV),
  distribute the init sibling, then split the i-strip into a MAIN (k=ii+mr-1..N,
  uniform across the strip, k pushed innermost → register-blockable) + a scalar
  CORNER (k=ii+i'..ii+mr-1). **trmm 1.0× → 2.38×** (broadcast win), checksum
  MATCH, no regression. lit `trmm-reduction-peel.mlir`.
- **(c) Explicit reduction-vec for the dot family — investigated, NO upside.**
  syr2k is already well-vectorized (133 vector / 45 scalar FMAs) and sits at the
  dot-register-block ceiling (1.63×, matching the explicit-vector hand-spike). Its
  limit is **register pressure** (4 input streams at the 2×2 tile), not
  vectorization quality — an explicit reduction-vec path would reproduce the same
  micro-kernel. Not worth a third codegen path. Deferred.
- **Remaining:** the SOTA study (BLIS / Polly / Bondhugula MLIR-GEMM).

## Reproduce
`/tmp/claude/gen/`: kernels `bmm_k.c ttm_k.c tcon2_k.c` (+ `_m.c` drivers),
tensor harness `runt.sh` (`one <name> <kern> <main> <LEAD>` — LEAD = dynamic
leading-dim size for the bare-ptr static-shape sed; tensors cubic so LEAD=128).
