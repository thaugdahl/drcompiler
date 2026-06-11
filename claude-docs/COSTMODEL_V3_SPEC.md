# Cost Model v3 — Stencils & Peeling: closing the remaining Polly gaps

**Date:** 2026-06-10 (evening). **Author:** Fable 5 implementation session.
**Audience:** a fresh implementation chat with no prior context.
**Predecessor:** COSTMODEL_V2_SPEC.md (fully implemented as checkpoints
`costmodel_v2_1`..`_8`; read its Ground Rules section first — repo, build,
checkpoint discipline, the cgeist -O0 mandate, and the SINK-checksum gate
all carry over verbatim).  Checkpoints for this spec: `costmodel_v3_N`.
**NEVER push.**

## 0. Where v2 ended (the baseline you must not regress)

Campaign `results/o0-campaign-2026-06-10` (cgeist -O0, 30/30 kernels, both
sizes, zero failures, all configs SINK-checksummed):

| geomean vs cgeist-base | LARGE | EXTRALARGE |
|---|---|---|
| distribute-tile-regblock (ours, best) | **1.69x** | **1.61x** |
| polly | 1.14x | 1.45x |

Ours beats Polly at both sizes; head-to-head 18/29 kernels at XL.  v3's
job is the remaining 11, which fall into exactly two families:

**Family A — peeling gaps (dense LA, Polly wins on shape coverage):**

| kernel (XL, vs base) | polly | ours | blocking issue |
|---|---|---|---|
| covariance | 6.25x | 3.70x | k-dim not tiled in the peeled HEAD (data columns re-streamed) |
| correlation | 1.90x | 1.02x | i-trip 2599 not mr-divisible → peel bails (A1) |
| trmm | 3.73x | 1.02x | undiagnosed; reduction-peel exists but doesn't fire (A3) |
| symm | 1.47x | 0.98x | scalar temp + scatter store shape (A4, hard) |
| gramschmidt | 2.25x | 0.94x | undiagnosed (A5) |
| lu | 1.03x | 0.86x | borderline; low priority |
| gemm (config artifact) | 2.41x | 1.81x | distribute-tile ALONE gets 4.45x; the combined config's regblock-before-tile order caps it (A6, cheap) |

**Family B — stencils (ours ~1.0x BY DESIGN; v2 only protected them):**

| kernel (XL, vs base) | polly | ours |
|---|---|---|
| seidel-2d | 2.27x | 1.09x |
| fdtd-2d | 1.34x | 1.00x |
| jacobi-1d | 0.34x (!) | 1.31x |
| heat-3d | 1.18x | 1.01x |
| jacobi-2d | 1.12x | 1.00x |
| adi | 1.09x | 1.13x |

Stencil upside is real but bounded (~1.1–2.3x each); Family A is worth
more geomean per unit of risk.  Do A before B.

## 1. Hard-won implementation lessons (read before touching anything)

These cost a full day to learn; each is load-bearing.

1. **The SINK gate catches everything — use it after every change.**
   v2 caught five distinct miscompiles this way (fusion slice erasure,
   in-place re-execution, lu interleaving, the empty-DIAG jam, the
   imperfect-body vectorizer).  Strict-FP (-O1) reference vs `none`;
   bit-identical required, except dot-family reassoc configs (2-ulp
   tolerance, see costmodel_v2_8 commit message).
2. **Array-dump diffing localizes what checksums only detect.**  Build at
   SMALL with `-DPOLYBENCH_DUMP_ARRAYS`, WITHOUT nodce.h (cgeist handles
   the fprintf dump at -O0), diff per-element vs `none`.  The error
   PATTERN names the bug: "rows scaled by a constant" → uncentered data →
   mean nest; "every 16th element right" → per-VL-lane statement
   execution.  Scripts: /tmp/fuseinv/covariance/mini2.sh + mini_link.sh.
3. **Stage bisection via knobs, not code edits:** `vectorize=false`
   isolates the explicit vectorizers; an mr that doesn't divide the trip
   (e.g. mr=7) silently disables peel+jam paths.  Two binary searches
   found the v2 vectorizer bug in four builds.
4. **Stage 3's unroll-jam is the resident hazard of AffineRegisterBlock.**
   Any loop shape your transform emits will be re-walked by the jam and
   the explicit vectorizers.  A bound dependence hidden behind an
   `affine.apply` looks jam-able and gets mangled (the empty-DIAG bug);
   the same nest with the bound directly on a loop IV is left alone.
   When emitting helper nests: original coordinates, bounds on real IVs.
   When in doubt, jam-proof the shape and verify with a SMALL dump diff.
5. **"It never ran" is a live hypothesis for any in-tree feature.**  The
   triangular peels sat under default-off `cache-tile` through every
   prior campaign.  Before extending a mechanism, prove the existing
   mechanism executes (instrument with temporary emitRemark, grep).
6. Infra: host `ninja dr-opt` for the edit loop; container only for
   cgeist/lowering/link (`/tmp/fuseinv/o0_build_any.sh <kernel> <variants>`,
   `gen_o0.sh` regenerates main_O0.mlir).  The lean-ctx shell wrapper
   mangles compound docker one-liners — put commands in script files.
   Campaign: `./run-campaign-v2.sh` (env DRCC_IMAGE=drcc-lean DRCC_TAG=
   x86_64); rebuild images with `./docker/build.sh --arch x86_64` after
   any dr-opt or tools/drcc script change.  Results dirs are archived,
   never overwritten.

## 2. Family A — peeling work packages

### A1. Remainder strips (unlocks correlation; trivial-ish, do first)

`peelTriangularNest` (AffineRegisterBlock.cpp:501) bails at :513 when
`(hi - lo) % mr != 0`; `peelTriangularReduction` (:656) likewise at :684.
correlation's corr nest has i ∈ [0, 2599) → never peels → 1.02x.

Fix: strip-mine only the divisible prefix and leave a scalar epilogue.
`stripHi = lo + ((hi - lo) / mr) * mr`; build the existing strip loop over
[lo, stripHi) and clone the ORIGINAL nest (original coordinates — lesson
4) with lower bound stripHi as the epilogue.  The epilogue is < mr rows of
a triangular nest: negligible time, zero risk.  Apply to both peels.

Acceptance: correlation XL ≥ 1.5x vs base (its corr nest is covariance's
shape; expect ~3x like covariance), SINK identical; covariance/syrk
numbers unchanged; a lit case with trip = N*mr + r (r ≠ 0) pinning
strip + epilogue structure.

### A2. k-tiling the peeled HEAD (covariance 3.7x → toward 6.25x)

The peeled+blocked HEAD streams full data COLUMNS per (i-strip, j): k is
untiled, so the two `data[k][..]` streams re-read N×M bytes per row-strip
(62 MB at XL — every strip).  Polly tiles k as well; that's most of its
remaining covariance lead.

Approach: after the peel produces the rectangular HEAD
`for i' { for j = ii+mr..M { for k = 0..N } }`, strip-mine k by Tk chosen
so the k-tile of both data column blocks fits L2 (reuse the v2 grid-search
machinery, or fixed Tk = l2Size/2 / (2·mr·8) rounded — measure both).
Two placements to A/B: (a) inside the peel (emit `for kk` around the
HEAD), (b) generic: teach dr-affine-loop-tile to tile bands whose root is
the strip loop (post-reg-block IR has vector ops → v2 tiler REJECTs as
out-of-model; a targeted "tile only loop k of this band" entry point
avoids that).  (a) is less general but jam-safe and recommended.

Acceptance: covariance XL ≥ 5x vs base, SINK identical, syrk/syr2k not
regressed (their k working set may already be resident — gate the k-strip
on footprint, don't apply unconditionally).

### A3. trmm diagnosis (reduction peel exists but 1.02x)

trmm kernel: `for i { for j { for k = i+1..M: B[i][j] += A[k][i]*B[k][j];
B[i][j] *= alpha } }`.  peelTriangularReduction (:656) targets exactly
this k = f(i)..N shape (MAIN/CORNER split, :650-653) — yet trmm measures
1.02x.  Suspects, in checking order:
  1. The trailing `B[i][j] *= alpha` statement makes the j-body imperfect
     — does the peel's candidate walk require onlyChildFor?  If so, the
     v2 statement-fission in dr-affine-loop-distribute should split it
     first… but fission legality may refuse: the scale statement and the
     k-loop both touch B, and the k-loop READS B[k][j] for k > i — rows
     not yet scaled.  Splitting "all accumulate, then all scale" vs the
     original interleave is a REAL dependence question (check what
     checkMemrefAccessDependence says at depth j; the in-place
     B-update may make fission illegal, which would be correct).
  2. If fission is illegal: peel must tolerate the trailing statement
     (keep it per-(i,j) in both MAIN and CORNER copies… only sound if it
     stays exactly once per (i,j) — put it in CORNER only, since CORNER
     runs last per row).  Design carefully against lesson 4.
  3. detectFamily on trmm: A[k][i] is k-strided (column), B[k][j] —
     j-stride-1 → Broadcast.  Check whether the MAIN actually
     register-blocks after peeling or bails later (instrument).
Deliverable: diagnosis note in the commit + fix if shape (2) is sound;
acceptance trmm ≥ 2x else a documented impossibility argument.

### A4. symm (hard; timebox or skip)

`for i for j { temp2 = 0 (scalar); for k < i { C[k][j] += alpha*B[i][j]
*A[i][k]; temp2 += B[k][j]*A[i][k]; }; C[i][j] = beta*C[i][j] +
alpha*B[i][j]*A[i][i] + alpha*temp2 }`.  Three obstructions at once: a
scalar loop-local accumulator (not a memref — our Acc matcher won't see
it), a SCATTER store C[k][j] varying in the reduction dim, and the
combined epilogue statement.  Polly only gets 1.47x here.  Recommended:
skip unless A1–A3+A6 land early; if attempted, the first step is scalar-
temp promotion to a 1-element memref so provenance/fission machinery can
reason about it, then split the k-loop into the scatter part and the
temp2 reduction (fission legality: the scatter writes C rows < i, the
epilogue writes C[i][j] — disjoint at depth j? check).

### A5. gramschmidt diagnosis (Polly 2.25x, ours 0.94x)

Untouched by v2 (sequential outer structure).  The inner norm/projection
loops (`nrm += A[i][k]²`, `R[k][j] = Σ A[i][k]·Q[i][j]`, update of A) are
column-major dot products — the same shape as covariance's k-nest.  The
0.94x (slightly WORSE than base) suggests one of our passes restructures
something unprofitably — diff `distribute-tile-regblock` IR vs `none` to
find which pass touches it, then either gate or exploit.  Deliverable:
diagnosis + at minimum restore 1.00x; upside if the dot nests vectorize.

### A6. Config-order tuning (gemm 1.81x → 4.45x; zero compiler work)

`distribute-tile-regblock` runs reg-block BEFORE the tiler; reg-block wins
on 2mm/3mm/syrk but caps gemm (its 4.45x comes from distribute-tile
alone).  Cheap experiment matrix on the harness side: add
`distribute-tile-then-regblock` (tile first) and per-order measurements on
the 8 dense-LA kernels; if no single order dominates, the decision is a
cost-model question: predict from ReuseAnalysis whether the tiled or the
register-blocked form wins (matmul-family detector: reg-block iff the
band matches the microkernel families AND trips divide; else tile).
Acceptance: a combined config ≥ max(distribute-tile, distribute-regblock)
- 5% on every dense-LA kernel.

## 3. Family B — stencils

Everything here is NEW capability (v2 only taught the cost models to say
no).  The prize list: seidel-2d 2.27x, fdtd-2d 1.34x, heat-3d 1.18x,
jacobi-2d 1.12x (Polly's numbers — treat as achievable targets, not
ceilings; note Polly LOSES jacobi-1d 0.34x, so its stencil machinery is
not uniformly good and neutrality must remain our floor).

### B1. Group reuse in ReuseAnalysis (prerequisite, analysis only)

ReuseAnalysis treats `A[t%2][i][j-1]` and `A[t%2][i][j+1]` as unrelated
references; stencils therefore show "no temporal reuse" and every gate
says REJECT — correct conclusion today, but blind: it cannot price
time-tiling.  Add reference GROUPS: same memref, same linear coefficient
matrix, differing only in the constant offset vector.  Per group:
  - groupSpan(dim) = max offset − min offset (the halo width);
  - group temporal reuse carried by loop ℓ when member offsets differ
    along a dim ℓ indexes (A[i-1], A[i+1] reuse each other's lines one
    i-iteration apart): reuse distance = footprint of `span` iterations
    of ℓ, not one.
  - PolyBench stencils at -O0 use `t % 2` ping-pong subscripts on some
    kernels — the v2 analyzer rejects mod as semi-affine.  Extend
    collectLinearCoeffs to tolerate `IV mod 2` in a LEADING subscript by
    treating it as a 2-valued select dimension (footprint ×2, no reuse
    classification on that dim), or pre-normalize: jacobi-2d/heat-3d at
    O0 emit two distinct memrefs (A,B) — check actual IR first; the mod
    case may only matter for adi/fdtd.  Lit: classify jacobi-2d's 5-point
    body (group of 5, span 1/1) and heat-3d's 7-point.
Deliverable: printer-pass output + lit pinning groups; no transform
changes; tiler/distribute behavior must be bit-identical after this lands
(their gates consume the same booleans — verify on the 10 staged
kernels).

### B2. Time-tiling for ping-pong stencils (jacobi-2d, heat-3d, fdtd-2d)

Shape: `for t { nest1: B ← f(A); nest2: A ← g(B) }` (already two perfect
inner nests; fission of t is illegal — v2's distribute correctly
refuses).  The classical transform: strip-mine t by Tt, skew the space
loops by the dependence distance per time step, tile space, so a
(Tt × Ts^d) tile's working set stays cache-resident across Tt steps.

Implementation plan (new pass `dr-affine-stencil-time-tile`,
func::FuncOp):
  1. MATCH: outer t-loop whose body is 1–2 perfect space nests; all
     dependence components between/within nests have |distance| ≤ 1 in
     every space dim per t step (extract via checkMemrefAccessDependence
     with dependenceComponents at each depth; this bound = skew factor 1).
     Constant trip counts everywhere (EXTRALARGE holds).
  2. TRANSFORM (start 1-D space = jacobi-1d, then 2-D):
     `for tt step Tt { for ii step Ti (skewed bounds i ∈
     [max(lo, ii - t·1), min(hi, ii + Ti - t·1)) expressed as affine
     min/max of (tt, t) ) { for t' { nest bodies with i-range clipped } }`.
     Affine min/max maps express the trapezoids; no ISL needed.  Build
     bounds programmatically as in peelTriangularNest (real IVs in maps —
     lesson 4).  Two inner nests stay sequential inside the tile (their
     inter-nest dependence is loop-independent per t).
  3. PROFITABILITY (ReuseAnalysis + B1): time-tile iff the per-step space
     footprint > LLC share (heat-3d XL: 27 MB ✓, jacobi-2d: 125 MB ✓) and
     trips known.  Tt from cache: Tt ≈ Ts·(line)/(halo traffic) — start
     with fixed Tt=8..32 swept manually, then encode what measures best.
  4. VALIDATION: time-tiling does not reorder the FP ops of any single
     point update (each (t,i,j) body executes whole) → SINK must be
     BIT-IDENTICAL.  MINI/SMALL dump-diff for first bring-up (lesson 2).
     Expect wrong-tile-bounds bugs to show as border-band corruption.
  5. Pipeline placement: its own config first
     (`func.func(dr-affine-stencil-time-tile)`), composed later.
Acceptance: jacobi-2d XL ≥ 1.15x, heat-3d ≥ 1.15x, fdtd-2d ≥ 1.2x vs
base, all bit-identical SINK, all dense-LA configs untouched.  fdtd-2d
has 4 inner nests over 3 arrays + a 1-D update — match generalization
needed; defer it to last.

### B3. seidel-2d (in-place Gauss–Seidel; biggest stencil prize, hardest)

Dependences within a time step: (1,0) and (0,1) and (1,1) AND the t-carried
(1,-1,-1)-ish family — space loops are NOT parallel, so B2's tiling shape
is illegal.  The classical answer is full skewing (t+i+j wavefronts) —
i.e. the one kernel where you genuinely need a polyhedral schedule.
Options: (a) hand-derive the legal skew for exactly this dependence
pattern (it is textbook: skew i by t, j by t+i; document the legality
argument from the extracted dependence vectors and let
checkMemrefAccessDependence verify on the transformed IR); (b) skip and
accept 1.09x.  Recommendation: attempt (a) only after B2 lands on two
kernels and only if time remains; Polly's 2.27x shows the ceiling.

### B4. adi (line solver; ours already 1.13x ≥ Polly 1.09x)

Leave alone.  Tridiagonal sweeps are sequential per line; nothing here
without algorithm-level changes.

## 4. Sequencing & budget

Order: A1 (small) → A6 (harness-only) → A2 → A3 → B1 → B2(jacobi-1d/2d
→ heat-3d → fdtd-2d) → A5 → B3/A4 (timeboxed).  After each A-package:
single-kernel host validation only (lesson 6); rebuild the image and
re-run the campaign ONCE at the end (or when the user asks).  Keep the
v2 protections absolute: any kernel at < 0.97x vs `none` in a config that
previously had ≥ 1.0x is a stop-the-line regression.

## 5. Non-goals

- Diamond/hexagonal tiling, ISL, or any general polyhedral scheduler —
  B2's trapezoid skew-tiling is the v3 ceiling for stencils.
- OpenMP/parallel codegen (everything stays single-thread).
- doitgen (pre-existing pipeline failure, unrelated to cost models —
  worth one diagnosis pass, tracked separately).
- Autotuning; all sizes/factors stay analytic or fixed-with-rationale.
