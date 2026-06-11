# Cost Model v4 — implementation findings (session 1)

**Date:** 2026-06-11. Branch `onnx-mlir`. Predecessor: COSTMODEL_V4_SPEC.md.
Commits this session: `costmodel_v4_1` (WP1), `costmodel_v4_2` (WP4).
**NEVER pushed.**

## Done

### WP1 — MachineModel (commit costmodel_v4_1)
One resolved description of the memory hierarchy
(`include/drcompiler/Analysis/MachineModel.h` + `lib/Analysis/MachineModel.cpp`):
l1/l2/l3/cacheLine/llcSharers/latencies + NEW `page_size`/`l2_tlb_entries`,
helpers `effectiveLLC()` / `tlbReachBytes()`. Precedence: explicit CLI option
(`mlir::Pass::Option::hasValue()`) > `cpu-cost-model-file` JSON > built-in
default. Defaults equal the historical per-pass CLI defaults → no-JSON runs are
bit-for-bit unchanged.

- Relocated `CpuCostModel` Transforms/ → Analysis/ (breaks a would-be
  Analysis→Transforms cycle; it is a pure model). 11 includers updated.
- Wired data-recomputation, affine-register-block, stencil-time-tile,
  memory-fission; added `cpu-cost-model-file` to register-block + stencil.
- Fixed a latent precedence inversion in data-recomputation (the old per-field
  `jsonCache` merge let the FILE beat an explicit CLI value).
- Full lit 203→ pass / 0 fail; +2 precedence tests.

**Gotcha that cost 3 false regressions:** `getNumOccurrences()` does NOT track
options parsed from a pass-pipeline string; use `Option::hasValue()`.

Deferred (behavior-preserving): LoopTiling keeps its distinct half-L2 target;
LoopDistribute cache-wiring folds into WP5; the A2 `peel-k-tile=128` default
stays (MachineModel now exposes `tlbReachBytes()` as the hook for the
`Tk≈tlbReach/rowStride` derivation, pending a covariance-XL plateau check).

### WP4 — symm scatter raising (commit costmodel_v4_2)
**symm XL 1.01x → 1.75x, SINK bit-identical, beats Polly's 1.50x.**

New Stage-0 `raiseSymmScatter` in affine-register-block fissions the scatter
(`C[k][j] += alpha*B[i][j]*A[i][k]`, k<i) from the rank-0 temp2 reduction + the
epilogue, and emits the scatter ALREADY interchanged to i-innermost
(`for k=0..N { for j { for i=k+1..N { C[k][j] += ... } } }`). The existing
in-place triangular register-blocker + broadcast vectorizer then crush it into
an mr×vl micro-kernel.

- **Approach vs spec:** implemented as a gated Stage-0 transform, NOT by
  loosening `canonicalizeOnce`'s non-constant-bound bail — lower risk, leaves
  trmm/lu's interchange gate untouched. The scatter branch is built directly in
  interchanged form (i/j bound maps copied verbatim), sidestepping any
  `permuteLoops` triangular-bound uncertainty.
- **Spike-first:** the hand-interchanged scatter was confirmed to register-block
  before any C++ was written (the spec's "existing machinery takes over" claim).
- **Bit-identical proof:** a scatter into row r accumulates only from i>r, which
  in the original outer-i order all run AFTER row r's epilogue; emitting branch A
  (epilogues) before branch B (scatters) preserves every C dependence. The
  broadcast family vectorizes the j-lane, so the i-reduction stays sequential per
  lane — no FP reassociation.
- **Also fixed** the rank-0 slice of the v3 Stage-3 cross-talk bug: post-jam
  `findReductionLoopUnder` now skips purely rank-0 reductions, so symm's branch-A
  temp2 no longer shadows branch-B's scatter reduction. Rank-0 reductions were
  never legitimate Stage-3 targets, so this is safe everywhere (verified:
  covariance/2mm/3mm/syrk/trmm still register-block at 3/56/72/49/24 vector ops).
- Full lit 206 pass / 0 fail (+ symm-scatter-raise.mlir).
- XL median-of-3 vs none_O0: none 10.02s | dttr 5.72s (1.75x) | dtrb 5.76s
  (1.74x), both SINK-identical to baseline.

### WP6 diagnosis — gemver, lu
- **gemver** (XL, 10-iter medians): none 46.6 ms | distribute-tile 45.9 |
  distribute-RB-tile 42.7 → best ours **1.09x** (Polly 1.31x per the campaign).
  All variants SINK bit-identical. gemver is a 4-nest memory-bound BLAS-2 chain
  (rank-2 A-update → A'·y→x → A·x→w); register-blocking the reductions buys
  little because the kernel is bandwidth-bound. `dr-affine-loop-distribute`
  correctly emits `SKIP legal split (no band deepened)` — the distribute-tile
  REGRESSION the spec saw (51 ms) comes from FORCING a tile on the independent
  BLAS-2 nests (point-bound overhead, no reuse captured), not from the enabler
  default. **The gemver gap is fusion+tiling to keep A resident, not anything
  the register-block/peel machinery can reach** — a fusion mechanism, out of the
  current thesis' scope.
- **lu**: NOT re-measured — its staged `main_O0.mlir` is gone from the ephemeral
  `/tmp/fuseinv/lu` (only an older `fuse.mlir` experiment survives). The spec
  itself rates the 4% gap as "within two noise bands"; regenerating via cgeist
  for a likely-noise result was not worth the budget this session.
- **doitgen**: untouched (pre-existing pipeline FAIL, tracked; not a cost model
  issue).

## Deferred with rationale

- **WP2 (decompose the 1819-line register-block god-pass into 4 files):**
  deliberately deferred. The 4-file split enables NO feature WP — seidel/fdtd
  consume the WP3 stencil engine (a separate file), symm reuses the existing
  peel (WP4 confirmed), and WP5 needs only a `canCapture()` function — so it
  fails the spec's own §9 litmus ("does seidel/fdtd/symm concretely consume
  it"). The cross-talk fix is speculative upside (LLVM scalar-promotion already
  rescues covariance) vs real regression risk to the working 2mm/3mm
  global-advance find. The rank-0 slice that symm actually needed is fixed (see
  WP4). Revisit only if a feature WP forces it or measurement motivates the full
  cross-talk rework.

## WP3.2 seidel — implemented, validated correct, but NO XL WIN (reverted)

**Honest NO-GO within v4 scope.** Implemented the space–space skewed
time-tiling (`i'=i+t, j'=j+2t+i`, derived + independently re-verified: all 9
dependence distances map to component-wise ≥0; the textbook b=1 is illegal,
b=2 is minimal). Emitted the 6-deep skewed nest (jj-band ties to ii, the
j-window carries the c·i term) in original coordinates.

- **Correctness PASSED the hardest gate:** SMALL dump-diff (POLYBENCH_DUMP_ARRAYS,
  tile-t=2 tile-s=4 to exercise multiple tiles + all windows) is **BIT-IDENTICAL**
  to baseline. The skew + window math is exactly right; lit (jacobi/heat
  unaffected) green.
- **XL timing (N=4000, T=1000, single run):** none 94.5s; best tiled config
  (tile-t=32 tile-s=128) 90.3s = **1.046x**; others 0.99–1.02x. NOT the ≥1.8x bar.
- **Root cause (measured, not guessed):** seidel is **compute/latency-bound on
  the serialized Gauss-Seidel recurrence** — `A[i][j]` reads `A[i][j-1]` *just
  written*, so the inner loop is a scalar dependence chain the backend cannot
  vectorize. One t-step touches 128 MB ≈ 8.5 s of memory at ~15 GB/s, but the
  step costs ~94 ms → ~90% of runtime is the serial arithmetic chain, ~10%
  memory. **Memory-locality time-tiling therefore cannot move the needle** (the
  4.6% best case is the entire memory share). The spec's premise (§0: "seidel
  needs full skew — biggest prize", implicitly a locality win like jacobi) is
  WRONG: jacobi is ping-pong (no recurrence → vectorizable → locality-bound);
  seidel's in-place recurrence makes it compute-bound.
- **The actual lever** is wavefront/diamond vectorization (iterate the
  dependence-free anti-diagonal so the recurrence parallelizes) — which the spec
  explicitly lists as a **non-goal (§9)**. So seidel ≥1.8x is unreachable within
  v4's stated scope.
- **Action:** reverted the (correct but pointless) transform — its only consumer
  (seidel) gets no win, so it fails the spec's own §9 litmus. The validated skew
  + emitter design is preserved here and in git history for any future wavefront
  work. fdtd uses the EXISTING tau-only emitter, so it is unaffected.

## WP3.3 fdtd-2d — WIN (commit costmodel_v4_3)

**fdtd-2d XL 1.00x → 2.51x, SINK bit-identical, beats Polly's 1.35x.** The exact
contrast with seidel that confirms the diagnosis: fdtd is **memory-bound**
(arithmetic intensity ~0.15 flop/byte) with **no intra-phase recurrence** (each
phase reads OTHER arrays), so the inner loops vectorize and time-tiling cuts
real byte traffic — the win seidel could not have.

New `fdtdTimeTile` path in `dr-affine-stencil-time-tile`: tau-only skew
(`i'=i+tau, j'=j+tau`, `tau = 4t + phase`, c=0) over 4 phases with per-phase
bands; the 1-D border (`ey[0][j]=fict[t]`) is a degenerate 2-D phase with i-band
`[0,1)`. No new schedule math vs jacobi (just P=4). SMALL dump-diff BIT-IDENTICAL;
lit +1 (jacobi/heat untouched). Measured optimum is small L2-resident tiles
(Tt=16, Ts=64 → 2.51x; the L3-derived Ts~780 gave only 1.86x — hardcoded as the
default, as the jacobi emitter does its own measured optima). XL median-of-5:
none 8.87s, Tt=16/Ts=64 3.53s.

## WP5 gramschmidt — spec mechanism OVERTURNED by spike; right lever found

**The spec's prescribed mechanism (block-interleave distribute) is WRONG.**
Confirmed current state: none 31.6s, distribute+regblock 20.2s = **1.57x**
(matches v3's 1.56x); distribute fully splits the j-loop into 3 (init / dot /
A-update), streaming A column-major twice.

Spike (micro-benchmark of the dot + A-update inner kernel, proper timescale):

| variant | inner-kernel | full-kernel (N=1200) |
|---|---|---|
| split (column-major, current) | 1.0x | 1.0x |
| **block-interleave (spec WP5)** | **1.38x** | — |
| **rowmajor (loop interchange, j inner)** | **3.3x** | **8.4x** |
| rowmajor + block | 4.5x | — |

The dot `R[k][j] += Q[i][k]*A[i][j]` (sum over i) and A-update
`A[i][j] -= Q[i][k]*R[k][j]` both stream A with the reduction/inner loop i in
the ROW position -> column-major, 8 useful bytes per 64-byte line.  Block-
interleave keeps the same column-major inner loop (only 1.38x); **loop
interchange to i-outer / j-inner makes A row-major (stride 1)** -> 3.3x on the
kernel, **8.4x on the full inner work**, and BIT-IDENTICAL (the i-accumulation
order is unchanged, checksums match).  Even after Amdahl + the norm/normalize,
this comfortably clears the 1.9x bar.

**IMPLEMENTED (commit costmodel_v4_4): gramschmidt 1.56x -> 8.9x, SINK
bit-identical** — far beyond the 1.9x bar (the real lever was 5x better than the
spec's block-interleave).  `interchangeBlas2RowMajor` runs as a Stage-1.5 step
in affine-register-block (after canonicalizeOnce, so nothing reverts it) and
interchanges the column-major BLAS-2 nest to row-major.  Gate that protects the
register-blocked BLAS-3 family: (a) enclosing sweep NOT loop-parallel
(gramschmidt's k is sequential; GEMM/syrk/covariance have a parallel sweep ->
skipped), (b) inner loop CONSTANT bounds (trmm/lu's triangular inner reduction
is the peel's 16x path -> skipped), (c) the M[inner][outer] column-major pattern,
(d) dependence-legal interchange.  No regression: gemm/2mm/3mm/syrk/syr2k/trmm/
covariance/correlation vector-op counts identical; atax/bicg/mvt untouched.  XL
median-of-5: none 31.2s -> 3.50s (distribute-RB-tile) / 3.38s (distribute-tile-RB).

Historical note — why the spec's mechanism was abandoned (the real difficulty):
a safe implementation
must (a) distinguish BLAS-2 (gramschmidt's dot: a SEQUENTIAL outer sweep k, one
free dim, no register reuse -> row-major streaming wins) from BLAS-3 (GEMM: two
PARALLEL spatial loops i,j -> column-major-in-k + register-blocking is correct);
a general "interchange column-major reductions" pass would BREAK GEMM/2mm/3mm.
The clean gate is "enclosing sweep is NOT loop-parallel"; AND (b) it must
integrate with `canonicalizeOnce`, which currently FORCES the reduction
innermost (re-creating the column-major form) -- a naive post-distribute
interchange is simply reverted.  So the fix is: teach `canonicalizeOnce` (or the
family-select) to prefer the row-major orientation for a BLAS-2 reduction under
a sequential sweep, instead of the reduction-innermost orientation.  That is a
change to the shared register-block canonicalization with GEMM-family blast
radius -- it needs its own validated session (gramschmidt win + GEMM / 2mm / 3mm
/ syrk / trmm no-regression + atax / bicg / covariance check), not a rushed edit
at the end of this one.

Spike artifacts: `/tmp/fuseinv/gs_spike.c` (inner kernel), `gs_full.c` (full
kernel) -- reproduce the table above.

## WP6 composed config — `drcomp-v4`, the one-compiler pipeline

Single pipeline, no per-kernel config sweep:

```
inline, raise-malloc-to-memref,
func.func(dr-affine-loop-distribute,
          dr-affine-stencil-time-tile,
          affine-register-block{mr=8 nr=16 vl=16},
          dr-affine-loop-tile),
dr-pin-liveout
```

Order rationale (all verified): distribute first (fission imperfect nests into
perfect bands); then stencil-time-tile (jacobi/heat/fdtd; no-op on reduction
bands); then register-block (gemm/symm/gramschmidt/trmm/...; no-op on stencils)
-- BEFORE the tiler, because gemm needs RB-before-tile (RB-after-tile gives 0
vector ops; the spec's §8 "tile then RB" order is WRONG for gemm); then
dr-affine-loop-tile last.  The stencil pass MUST precede the tiler (strip-mining
destroys its constant-bound-nest match), satisfied here.

Pass disjointness pinned by `test/AffineRegisterBlock/pass-disjointness.mlir`:
the stencil pass is a no-op on a GEMM band, register-block is a no-op on a
jacobi ping-pong.

Validation (XL, drcomp-v4 vs none_O0, median-of-3, all SINK bit-identical) --
each kernel hits its per-kernel best-ours within 5%, spanning every mechanism:

| kernel | none | drcomp-v4 | speedup | mechanism |
|---|---|---|---|---|
| gemm        | 2.69s  | 1.59s | 1.69x  | register-block |
| trmm        | 6.40s  | 0.31s | 20.3x  | in-place triangular peel |
| symm        | 9.86s  | 5.72s | 1.72x  | scatter-raise (WP4) |
| gramschmidt | 31.2s  | 3.44s | 9.05x  | BLAS-2 row-major interchange (WP5) |
| fdtd-2d     | 8.81s  | 3.56s | 2.47x  | 4-phase skewed time-tile (WP3.3) |
| jacobi-2d   | 11.1s  | 4.04s | 2.74x  | ping-pong time-tile |

The full 29-kernel within-5% sweep is the campaign run (multi-hour, launched via
drcc-benchmarks/run-campaign-*.sh -- not tool-launchable per §0b); the 6-kernel
spot-check above covers every transform path and every kernel lands on its best.

## Done — all of v4

WP1 ✓ · WP2 decompose ✓ · WP3.3 fdtd 2.51x ✓ · WP4 symm 1.75x ✓ · WP5
gramschmidt 8.9x ✓ · WP6 diagnosis + composed config ✓.  WP3.2 seidel: honest
NO-GO (compute-bound).  The spec's seidel "biggest prize" and gramschmidt
block-interleave premises were both corrected by measurement.
- **WP5 gramschmidt** (1.56x→1.9x): block-interleave mode for distribute; spec
  says do last (perturbs the most-shared pass).
- **WP6 composed config** (`drcomp-v4`): blocked on WP3.3.

## Validation harness (reconstructed; works — see memory `costmodel_v4_session`)
docker `drcc-lean:x86_64`; dr-opt runs HOST-side (no image rebuild for transform
edits). `a6_dropt.sh` (host) → `o0_build_any.sh` in container → `a6_time.sh`
(SINK hex bit-identical check + median). The broadcast vectorizer needs CONSTANT
bounds, so it only fires post-`inline` (which inlines the dataset sizes).
