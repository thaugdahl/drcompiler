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

## Not started (large feature work — next session)

- **WP3.1/3.2 seidel** (the biggest prize, 1.09x→≥1.8x): needs the stencil
  engine generalized to space–space skew (`i'=i+t, j'=j+2t+i`; the existing
  emitter is tau-only with symmetric bands — substantial new code) AND the
  jacobi/heat bit-identical refactor bar. Highest miscompile risk of v4
  (in-place + skewing) — dump-diff FIRST. Staged kernel present
  (`/tmp/fuseinv/seidel-2d`, IR confirmed to match spec §4.2).
- **WP3.3 fdtd** (≥1.2x): 4-phase, needs per-phase bands; tau-only suffices
  (f=1,c=0). After seidel.
- **WP5 gramschmidt** (1.56x→1.9x): block-interleave mode for distribute; spec
  says do last (perturbs the most-shared pass).
- **WP6 composed config** (`drcomp-v4`): blocked on WP3.3.

## Validation harness (reconstructed; works — see memory `costmodel_v4_session`)
docker `drcc-lean:x86_64`; dr-opt runs HOST-side (no image rebuild for transform
edits). `a6_dropt.sh` (host) → `o0_build_any.sh` in container → `a6_time.sh`
(SINK hex bit-identical check + median). The broadcast vectorizer needs CONSTANT
bounds, so it only fires post-`inline` (which inlines the dataset sizes).
