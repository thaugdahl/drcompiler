# PolyBench family sweep — register-block generalization + safety boundary

**Date:** 2026-06-05
**Author:** Claude (Opus 4.8)
**Ask (user):** Implement family auto-selection in the pass; then investigate a
larger PolyBench set and map which *families* register-blocking generalizes to.

**Result:** The pass now **auto-selects** the register-block configuration from
operand layout (no manual flags) and **refuses to transform anything unsafe**.
Across a 9-kernel PolyBench sweep it wins **1.6–2.6×** on every contraction
kernel and is neutral/correct on the rest. The sweep also exposed — and the pass
now guards against — a **correctness bug** (LU) and a **regression** (trmm).
All numbers N=1024, Ryzen 7950X3D core 8, vs `clang -O3 -march=native -ffast-math`
(best loop order), checksum-verified (`correct=MATCH` = rb output == clang output).

---

## Measured (single default pass invocation: `affine-register-block{cache-tile=true ...}`)

| kernel | class | fires | correct | rb/clang | auto-config |
|---|---|---|---|---|---|
| gemm       | broadcast        | yes | MATCH | **2.32×** | 8×16, no-reassoc |
| 2mm        | broadcast        | yes | MATCH | **2.31×** | 8×16, no-reassoc |
| 3mm        | broadcast        | yes | MATCH | **2.39×** | 8×16, no-reassoc |
| covariance | broadcast (gram) | yes | MATCH | **2.55×** | 8×16, no-reassoc |
| syrk       | dot (rank-k)     | yes | MATCH | **2.63×** | 4×4, **reassoc** |
| syr2k      | dot (rank-k)     | yes | MATCH | **1.64×** | 2×2, **reassoc** |
| trmm       | triangular bcast | no  | MATCH | 1.00× (neutral) | peel-limit, untouched |
| lu         | factorization    | no  | MATCH | — (untouched)   | alias-guarded off |
| gramschmidt| factorization    | scalar only | MATCH | 1.00× | dep-guarded off |

## The family taxonomy (the deliverable)

Register-blocking applies to a **2D-spatial × 1D-reduction contraction nest with a
reduction-invariant accumulator**. Within that, *operand layout* (not the math)
picks the configuration; outside it, the pass must decline.

**1. Broadcast contraction — `C[i][j] += a(i,k) · b(k,j)`, 2nd operand stride-1 in
the spatial `j`.** gemm, 2mm, 3mm, **covariance**, (doitgen, symm-core).
→ wide 8×16 tile, **no** reassociation; LLVM SLP-vectorizes the independent
`j`-lanes. **2.3–2.6×.**
*Key insight:* a gram matrix is **layout-dependent**. covariance computes
`covᵀcov`-style `Σ_k D[k][i]·D[k][j]` but with **row-major** data `D`, so `D[k][j]`
is stride-1 in `j` → **broadcast** (2.55×). Same `XᵀX` math as syrk, opposite
family — because syrk's `A[j][k]` is the *stored transpose* (stride-1 in `k`).

**2. Dot contraction (rank-k) — operands stride-1 in the reduction `k`.** syrk,
syr2k. → small square tile (syrk 4×4, syr2k 2×2 — shrinks with input-stream count),
**reassoc set** so LLVM reduction-vectorizes over `k`. **1.6–2.6×.**

**3. Triangular broadcast — broadcast contraction with an IV-dependent reduction
bound.** trmm (`k = i..N`). The diagonal peel currently handles the
*spatial* triangle (syrk's `j:0..i+1`) but not the *reduction* triangle, so trmm
is **left untouched (neutral 1.0×)** rather than mis-transformed. Clear next
extension (would turn it into a win).

**4. Factorization / solver — contraction *update* with a carried dependency.**
lu, ludcmp, cholesky, gramschmidt, trisolv, durbin. The inner update looks like a
reduction but (a) the accumulator **aliases** its inputs in place (LU's
`A[i][j] -= A[i][k]·A[k][j]`), and/or (b) the outer sweep **carries a dependency**
(gramschmidt updates `A` across `k`). Register-blocking is **illegal** here; the
pass declines (only the harmless scalar norm-reduction in gramschmidt is
SSA-promoted). Needs *blocked-factorization* algorithms — a different technique,
out of scope.

**5. Out of scope — no 2D contraction.** BLAS-2 matvec (gemver, gesummv, mvt,
atax, bicg): 1D output, memory-bound, no 2D register tile. Stencils (jacobi,
seidel, heat, fdtd, adi): no reduction contraction at all.

## Two bugs the larger sweep exposed (and the guards that fix them)

Both are *correctness/no-regression* guards added to the pass; neither touches the
winners (verified: all six still fire at full speed, 7/7 lit tests pass).

1. **LU correctness violation.** The detector saw `A[i][j] -= A[i][k]*A[k][j]` as
   a register-blockable reduction, but the accumulator `A` aliases its
   multiplicands → reordering changed results (`NAIVE≠RB≠CLANG`). **Fix:**
   `accumulatorAliasesInput` — reject if the body reads the accumulator's memref
   at an address that is never stored back here (a true read-only cross-access).
   A *sibling* accumulator (`C[i+1][j]` after unroll-jam, has a matching store) is
   correctly allowed — the first cut over-rejected these and broke gemm, fixed.
2. **trmm 50× regression.** The pass interchanged trmm to reduction-innermost
   expecting to block, but trmm's `k=i..N` reduction can't be cleanly jammed, so
   it failed to block and left a cache-hostile column-major order (0.02×).
   **Fix:** don't interchange a reduction whose trip count depends on an outer IV
   unless we can block it; leave it untouched (clang vectorizes the original).
3. **General safety net — parallelism guard.** Only unroll-and-jam loops that are
   `isLoopParallel`. gemm/syrk spatial loops are parallel; gramschmidt's carried
   `k` is not → its dangerous reduction is refused (only the scalar norm is
   promoted). This is the textbook correctness condition for unroll-and-jam and
   guarantees soundness beyond the specific kernels tested.

## Further generalization (which other families)

- **Tensor contractions / batched matmul / conv-as-matmul** (doitgen; and the
  DL-relevant ops — conv lowered to matmul, attention `QKᵀ`/`·V`). These are
  broadcast-family and should generalize; the only gap is the pass currently
  blocks a fixed 2D spatial pair — picking any 2 of N free dims would cover them.
  **Highest-value next step** for the ONNX-MLIR / MARCO memref-path scope.
- **Triangular broadcast (trmm, symm):** extend the diagonal peel to the
  reduction-bound triangle → trmm becomes a win. Moderate effort.

## Pass changes (lib/Transforms/AffineRegisterBlock.cpp, Passes.td, CMakeLists)
- `detectFamily` + `innermostStrideOne`: classify broadcast vs dot from operand
  stride; Stage 1a picks tile (8×16 / 4×4 / 2×2) and reassoc.
- Stage 4: set `arith.fastmath<fast>` on the reduction FP ops for the dot family
  (lowers to LLVM `fast` → backend vectorizes the k-reduction). `family-select`
  option (default on); `family-select=false` reproduces the raw transform.
- `accumulatorAliasesInput` guard (LU); IV-dependent-reduction-bound guard (trmm);
  `affine::isLoopParallel` guard (general). New lit tests `syrk-family-select.mlir`,
  `gemm-no-reassoc.mlir`; `syrk-triangular-peel.mlir` pinned to `family-select=false`.
- Full suite: 188 pass / 16 pre-existing-ONNX unresolved / **0 fail**.

## Reproduce
`/tmp/claude/gen/run_full.sh` (default family-select pass over all 9 kernels);
kernels `*_k.c` + drivers `*_m.c` in `/tmp/claude/gen/` (covar, trmm, lu, gram new).
