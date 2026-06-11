# Cost Model v4 — closing the remaining losses, paying the architecture debt

**Date:** 2026-06-11. **Author:** Fable 5 (v3 implementation session).
**Audience:** a fresh implementation chat with no prior context.
**Predecessors:** COSTMODEL_V2_SPEC.md (ground rules carry over verbatim:
repo, build, checkpoint discipline `costmodel_v4_N`, cgeist -O0 mandate,
SINK gate, **NEVER push**) and COSTMODEL_V3_FINDINGS.md (read the
"Diagnoses & oddities" section — every bug listed there is load-bearing
context for the refactors below).

## 0a. Session bootstrap (read before anything else)

- **Everything under `/tmp/fuseinv/` is EPHEMERAL** and may be gone.
  That includes the staged kernels, the verified-IR dumps cited in this
  spec (e.g. `/tmp/fuseinv/symm/base.mlir`), and all helper scripts.
  Recreate as needed; the load-bearing ones, in order of use:
  - `gen_o0.sh` — container-side cgeist -O0 → main_O0.mlir per kernel
    (cgeist flags + the vector<Nxi32>→i64 sed + rewrite-struct-memrefs;
    the exact recipe is reproduced in polybench-bench.sh's cgeist step).
  - `o0_build_any.sh` — container-side lower/link of a dr-opt'd .mlir
    into timing bin + strict-FP `_chk` bin (mlir-opt lowering flags are
    in COSTMODEL_V2_SPEC ground rules).
  - `a6_time.sh` — SINK-check + median-of-3 timing vs none_O0.
  - `mini_stencil.sh` / `mini_host*.sh` — SMALL + POLYBENCH_DUMP_ARRAYS
    dump-diff harness (lesson 2 of v3; build WITHOUT nodce.h at SMALL).
  Host dr-opt invocations must mirror the harness wrap:
  `--pass-pipeline="builtin.module(inline,<pipeline>,dr-pin-liveout)"`.
- **Repo state at handoff:** branch `onnx-mlir`, local checkpoints
  `costmodel_v2_1..8` + `costmodel_v3_1..6` + findings + this spec.
  NEVER push.  The drcc-benchmarks repo has UNCOMMITTED changes that the
  campaign depends on: polybench-bench.sh (configs
  `distribute-tile-then-regblock`, `stencil-time-tile`) and the
  untracked `run-campaign-v3.sh` — do not clobber or `git checkout --`
  them.
- **The container image bakes dr-opt**: after any dr-opt or
  tools/drcc change, `./docker/build.sh --arch x86_64` before any
  container-side build (host `ninja dr-opt` is enough for the host edit
  loop and lit).
- The lean-ctx shell wrapper mangles compound/multi-line shell —
  put anything non-trivial in a script file; `git commit -F <file>`.

## 0b. Tacit gotchas (learned by hitting them; not inferable from the code)

- **`ninja -C build check-drcompiler` exits 1 on a HEALTHY tree.**
  16 pre-existing ONNX intermediate tests are "Unresolved" (no RUN
  lines) and lit counts that as failure.  The real signal is the
  Passed/Failed lines (203 passed / 0 failed at v3 handoff).  Run
  `/usr/bin/lit -sv <dir>` from `build/test/` for targeted runs
  (`llvm-lit` is not on PATH on this host).  Do NOT chase the ONNX
  unresolveds; they predate every costmodel checkpoint.
- **Never compile while timing.**  `ninja dr-opt` and `docker build`
  saturate all cores; a benchmark running concurrently is garbage.
  Sequence: edit → build → THEN measure.  And distrust any per-kernel
  conclusion under ~50 ms total runtime (jacobi-1d XL is ~1 ms; its
  campaign ratios are pure noise — see V3_FINDINGS' stop-the-line
  analysis).
- **Container-side scripts must chown outputs**
  (`-e HOST_UID=$(id -u) -e HOST_GID=$(id -g)` + a trailing
  `chown ${HOST_UID}:${HOST_GID}` in the script, as all v3 scripts do).
  A docker run without it leaves root-owned files in the staging dir
  and every later host-side write fails with EACCES.
- **Multi-hour jobs (the campaign) cannot be tool-launched directly** —
  background shells are killed at the tool timeout.  Launch via a
  script that does `setsid nohup ./run-campaign-v3.sh > log 2>&1 &`,
  then poll/tail the log.
- **The peel drivers terminate by re-bail, not by bookkeeping.**  Each
  while-loop re-walks candidates after every successful peel; the
  artifacts a peel emits (the < mr epilogue, the CORNER, the MAIN) are
  themselves re-matched and must bail (`stripHi == lo`, non-constant
  bounds, missing in-place read).  If WP2's refactor or any new peel
  changes a bail condition, check termination on a kernel with a
  non-divisible trip — an infinite peel loop presents as dr-opt
  hanging, not crashing.
- **Stage 3b's `onlyChildFor(sIn) == red` gate is load-bearing AND a
  known limitation.**  It is why the post-jam triangular HEAD (whose j
  body holds 8 hoisted affine.applys) is NOT re-vectorized by 3b — that
  HEAD wins via LLVM scalar promotion + SLP and measures fine.  Do not
  "fix" 3b to accept imperfect bodies without re-deriving the
  every-VL-lanes soundness argument (the v2 mean-nest miscompile).
- **The campaign CSVs contain literal `FAIL` strings in cells**
  (doitgen row) — float-parse defensively when writing analysis
  scripts.
- **a6_time.sh-style SINK checks print DIFF for syrk/syr2k** — that is
  the documented 2-ulp dot-family reassoc tolerance (v2 ground rules),
  not a regression.  Everything else must be bit-identical.

## 0. Where v3 ended

Campaign `results/o0-campaign-2026-06-11-v3`: geomean 2.05x XL / 2.12x
LARGE vs cgeist-base (Polly 1.42x / 1.14x), head-to-head 21/29 at both
sizes, zero failures. The remaining XL losses:

| kernel | ours | polly | class |
|---|---|---|---|
| seidel-2d | 1.09x | 2.27x | needs full (t,i,j) skew — biggest prize |
| gramschmidt | 1.56x | 2.26x | distribute split costs refetch; dot is fast, update nest is not |
| symm | 1.01x | 1.50x | scalar temp + scatter store + epilogue statement |
| fdtd-2d | 1.00x | 1.35x | 4-nest/3-array stencil; v3 matcher only does ping-pong pairs |
| gemver | 1.17x | 1.31x | undiagnosed; fusion-shaped |
| lu | 1.00x | 1.04x | borderline; diagnosis-only |
| durbin / nussinov / floyd-warshall | ties | ties | no action |

Two structural observations from v3 that shape everything below:

1. **Every v3 win came from making a transform REACH a nest it already
   knew how to handle** (peel divisibility, alias certification, order of
   passes, parallel-pair requirement). The remaining losses are the
   cases where the *mechanism itself* is missing (seidel/fdtd skewing,
   symm shape) or where two passes *disagree about who profits*
   (gramschmidt). v4 is therefore part new mechanism, part architecture.
2. **The v3 codebase took on real debt to move fast.** It is documented
   honestly in §1 and paid down in WP1–WP2 *before* the new mechanisms,
   because seidel and fdtd would otherwise fork yet another copy of the
   skewing logic, and symm would add a fourth near-copy of the peel
   builder.

## 1. Architecture debt — what is wrong with the current structure

Read this as a critique of code I (the v3 session) wrote or extended;
none of it is hypothetical.

### 1.1 AffineRegisterBlock.cpp is a 1900-line god-pass

It currently contains: reduction-nest canonicalization (interchange),
family selection, THREE triangular peels (`peelTriangularNest`,
`peelTriangularReduction`, `peelInPlaceTriangularInnermost` — each a
near-copy of the same strip/epilogue/clone scaffolding), cache tiling,
the A2 k-chunker, unroll-jam orchestration, two explicit vectorizers,
the Stage-3b leftover sweep, and fastmath tagging. Consequences already
observed:

- **The Stage-3 cross-talk bug** (`findReductionLoopUnder(func)` at
  AffineRegisterBlock.cpp:1710 and :1744 after a
  jam picks up whichever innermost reduction comes first in walk order —
  on covariance it processes the *mean* nest and silently leaves the
  jammed cov HEAD unpromoted). The pass only works because LLVM's scalar
  promotion rescues the abandoned nest. This is a direct symptom of
  stages communicating through "whatever the IR looks like now" instead
  of explicit work items.
- **The `dr.acc_no_alias` magic attribute** is a cross-stage contract
  smuggled through the IR: the in-place peel proves a fact, encodes it as
  an attribute, and two helper functions 600 lines away know to look for
  it. It also silently *evaporates* when upstream utilities clone or
  reconstruct loops (we measured it surviving unroll-jam by luck of
  implementation, and being consumed by the vectorizer rebuild). A
  certification that can be dropped without an error is not a contract.
- **Peels communicate with Stage 3 by emitting "jam-proof shapes"**
  (lesson 4 of v3): bounds on real IVs, applies inside the innermost
  body, CORNERs in original coordinates so a downstream parallelism
  check *happens* to reject them. Nothing enforces any of this; the
  empty-DIAG miscompile of v2 was exactly this convention being broken.

### 1.2 The machine is described six times

`affine-register-block`, `dr-affine-stencil-time-tile`,
`dr-affine-loop-tile`, `dr-affine-loop-distribute`, `memory-fission`,
and `data-recomputation` each carry their own `l3-size`/`llc-sharers`
(and assorted l1/l2/latency) options with independent defaults. The v2
notes already record one drift incident (a stale l2 default 4x too small
on one path). Tonight added a new dimension: the A2 covariance win is a
**TLB/page-locality** effect, but no pass has page-size or TLB-reach
parameters — the knowledge is frozen into `peel-k-tile=128, floor 64`, a
magic number that encodes "about how many 4 KB pages a 1.5K-entry L2 TLB
can hold per slab" without saying so.

### 1.3 Profitability is guessed twice, differently

The gramschmidt 0.94x (fixed in v3_6) happened because
`dr-affine-loop-distribute`'s locality guard asks "will some downstream
pass capture reuse from this split?" by *re-deriving its own
approximation* of what the tiler/regblock will do — and stays optimistic
whenever its constant-coefficient model fails
(DrAffineLoopDistribute.cpp:323, the `reuseBenefit = true` on analysis
failure). The honest fix was not
applied in v3 (we compensated downstream with Stage 3b). The general
principle violated: **a pass that splits for the benefit of a downstream
pass must ask that pass, not impersonate it.**

### 1.4 The stencil pass hardcodes its schedule

`dr-affine-stencil-time-tile` bakes in: 2 phases, skew factor 1 per
half-step, halo ≤ 1, identical bands. The schedule derivation (virtual
time, skew amounts, window emission) is interleaved with the matcher.
seidel-2d needs a *different* skew (space–space, not just time–space)
and fdtd-2d needs N=4 phases over 3 arrays — implemented naively these
become two more forks of the same trapezoid-emission code.

### 1.5 Config combinatorics instead of composition

`distribute-tile-regblock`, `distribute-tile-then-regblock`, and
`stencil-time-tile` are separate harness configs; the campaign's
"best-ours" is a per-kernel max over configs — fine for research tables,
but it means there is no single pipeline that is *the* compiler. The A6
finding (tile-before-regblock ≥ everything) removed one axis; stencils
still live outside.

## 2. WP1 — `MachineModel`: one machine, one description (foundation)

New `include/drcompiler/Analysis/MachineModel.h` +
`lib/Analysis/MachineModel.cpp`.

```
struct MachineModel {
  int64_t l1Size, l2Size, l3Size, cacheLine;
  unsigned llcSharers;            // effective LLC = l3Size / sharers
  int64_t pageSize;               // NEW (4096)
  int64_t l2TlbEntries;           // NEW (~1536 typical Zen4/SKX)
  unsigned l1Lat, l2Lat, l3Lat, memLat;
  int64_t effectiveLLC() const;
  int64_t tlbReachBytes() const;  // pageSize * l2TlbEntries
  static MachineModel fromJson(StringRef path);  // cpu-cost-model.json
};
```

- Parsed once from the existing `cpu-cost-model-file` JSON (extend the
  schema with `page_size`, `l2_tlb_entries`; defaults preserve current
  behavior bit-for-bit).
- Every pass keeps its CLI options **as overrides only** (option set ⇒
  wins; unset ⇒ MachineModel value). This preserves all lit tests and
  harness configs while making the JSON the single source of truth.
- **A2's k-chunk target becomes derived**: `Tk_target ≈ tlbReach /
  (rowStrideBytes)` clamped to [64, 512] — re-derive, then VERIFY it
  still lands on the measured 75–150 plateau for covariance XL before
  switching the default; if it does not, keep 128 and document the model
  miss. (Analytic-with-rationale, per the v3 non-goals.)

Acceptance: zero behavior change with no JSON present (full lit + the
10 staged kernels' pipeline IR bit-identical); one lit test for JSON
override precedence.

## 3. WP2 — decompose AffineRegisterBlock (refactor, no behavior change)

Split into four files under `lib/Transforms/RegisterBlock/`:

1. **`ReductionMatch.{h,cpp}`** — the *only* place that recognizes
   accumulator patterns. Exposes
   `FailureOr<ReductionBand> matchReductionBand(AffineForOp innermost)`
   returning an explicit struct `{red, sIn, sOut, accs, family, flags}`.
   The alias guard lives here, *as a query with an explicit override
   parameter* — `matchReductionBand(loop, AliasPolicy::CertifiedDisjoint)`
   — which **replaces the `dr.acc_no_alias` attribute entirely**: the
   in-place peel calls the vectorizer directly on the MAIN it just built
   (it has the `ReductionBand` in hand; there is nothing to smuggle
   through the IR and nothing for upstream utilities to drop).
2. **`Peel.{h,cpp}`** — one `StripPeelBuilder` owning the shared
   scaffolding (divisible-prefix strip, original-coordinate epilogue,
   body cloning with IV remap) + three thin shape adapters (ub-tri,
   reduction-tri, in-place). The three current peels collapse onto it;
   their lit tests pin that the emitted IR is unchanged.
3. **`Vectorize.{h,cpp}`** — `vectorizeBroadcastBand`/`vectorizeDotBand`
   + the symbolic vl-split, taking a `ReductionBand` instead of raw
   loops.
4. **`AffineRegisterBlockPass.cpp`** — the driver: builds an explicit
   **worklist of `ReductionBand`s captured before any mutation**, then
   processes each (peel → jam → vectorize → promote). This kills the
   `findReductionLoopUnder(func)` re-scan and with it the cross-talk
   bug: after jamming a band, the driver re-matches *that band's
   location* (the jam invalidates handles, not identity — re-match from
   the stable parent), never "the first reduction anywhere".

Risk control: this WP changes ZERO intended behavior except the
cross-talk fix, which must be validated like a feature — covariance,
correlation, syrk, trmm, gemm, 2mm, 3mm, gramschmidt single-kernel SINK
+ timing before/after (the fix plausibly *improves* covariance: the cov
HEAD becomes explicitly vectorized instead of relying on LLVM scalar
promotion + SLP; it may also change nothing — measure, don't assume).
Acceptance: full lit green, dense-kernel SINKs identical, no kernel
< 0.97x of its pre-refactor time.

## 4. WP3 — skewed-tiling engine + seidel-2d (the prize)

### 4.1 Extract the engine from the v3 stencil pass

Split `dr-affine-stencil-time-tile` (matcher currently hardcodes
2-phase ping-pong at AffineStencilTimeTile.cpp:185) into:

- **`StencilSchedule.{h,cpp}` (lib/Analysis)** — given a time loop and a
  list of phase nests, extract dependence distance vectors (via
  `checkMemrefAccessDependence` with components, per phase pair) and
  derive a per-space-dim **skew vector**, not just a tau factor:
  `i'_k = i_k + f_k·tau + sum_{m<k} c_km·i_m`.  The space–space terms
  `c_km` are REQUIRED — seidel-2d is not schedulable with tau-only
  skewing (derivation in 4.2; the v3 engine's tau-only model is a
  special case with all c = 0, which is what jacobi/heat use).
  Validity = all skewed distances component-wise ≥ 0 (fully
  permutable); solve for the smallest (f, c) meeting it, bail if the
  dep cone needs anything beyond this affine form.  For the jacobi
  family it must derive exactly the v3 numbers: P=2, f=1, c=0
  (lit-pinned).
- **`SkewedTileEmitter.{h,cpp}` (lib/Transforms/Utils)** — given
  (time loop, phases with PER-PHASE space bands, P, skew vectors, Tt,
  Ts, MachineModel) emit the band-local tile loops + per-phase max/min
  windows.  Two generalizations over v3, both forced by real kernels:
  per-phase bounds (fdtd's phases have four DIFFERENT bands, see 4.3;
  v3 required identical bounds) and space–space window terms (windows
  become max/min over (tt, ii_k, t, outer i_m) — still plain affine
  maps on real IVs, no applies between loops).  Skew slack per dim is
  `f_k·P·Tt + sum c_km·Ts` — the cache-derived Ts formula must use it
  (v3 used 2·Tt, the P=2/f=1/c=0 case).
  The v3 pass becomes matcher + profitability gate + one emitter call.

Acceptance for the extraction alone: jacobi-1d/2d, heat-3d pipeline IR
bit-identical to v3_5 output (this is the bar that proves the refactor
is a refactor).

### 4.2 seidel-2d matcher

Verified IR (cgeist -O0, XL): single in-place 9-point nest, ONE memref,
`for t = 0..1000 { for i = 1..3999 { for j = 1..3999 {
A[i][j] = (9-point sum)/9 } } }`, N=4000 → 128 MB per step, footprint
gate trivially satisfied.  P=1 (tau = t).

Dependence set (store A[i][j]; loads at all (di,dj) ∈ {-1,0,1}²):
points lexicographically earlier in the same step were already
rewritten → same-t flow deps (0,1,{-1,0,1}) and (0,0,1); points later
in the step still hold the previous step's value → cross-t flow deps
(1,{-1,0},{-1,0,1}) (plus matching anti/output deps, all dominated).
Solving `Δi' = Δi + a·Δt ≥ 0`, `Δj' = Δj + b·Δt + c·Δi ≥ 0` over that
set: a ≥ 1; c ≥ 1 (from (0,1,-1)); b ≥ c+1 = 2 (from (1,-1,-1)).
**Minimal legal skew: i' = i + t, j' = j + 2t + i.**  NOTE: the v3
spec's B3 recollection ("skew i by t, j by t+i", i.e. b=1) is ILLEGAL —
the (1,-1,-1) dependence (reading A[i+1][j+1], last written at
(t-1, i+1, j+1)) gives Δj' = -1+1-1 = -1.  This is exactly why the
engine must DERIVE skews from extracted vectors and must re-verify the
transformed IR with `checkMemrefAccessDependence` in a debug mode;
textbook recall is not a legality argument.

Matcher gate: single nest, one memref, store offset 0, load offsets
within ±1, footprint gate as in ping-pong.

Validation: highest-miscompile-risk item of v4 (in-place + skewing).
MINI/SMALL dump-diff FIRST, then SINK at XL.  Gauss–Seidel is *defined*
by its in-place update order and the skewed schedule preserves the
original execution order of every point update exactly — dumps must be
BIT-identical, no tolerance.

Acceptance: seidel-2d XL ≥ 1.8x (Polly 2.27x is the reference, not the
ceiling), jacobi/heat numbers unchanged, fdtd-2d unaffected.

### 4.3 fdtd-2d matcher (after 4.2 lands)

Verified IR (cgeist -O0, XL): t = 0..1000 with FOUR phases over three
2000x2600 arrays + a 1-D source vector:

| p | writes | band | reads (offsets) |
|---|---|---|---|
| 0 | ey[0][j] | j ∈ [0,2600) (1-D border) | fict[t] |
| 1 | ey[i][j] | i ∈ [1,2000) × [0,2600) | hz (0,0), (-1,0) |
| 2 | ex[i][j] | [0,2000) × [1,2600) | hz (0,0), (0,-1) |
| 3 | hz[i][j] | [0,1999) × [0,2599) | ex (0,0),(0,+1); ey (0,0),(+1,0) |

All inter-phase dependences have tau-distance ≥ 2 at halo ≤ 1 (e.g.
p1→p3 same t: Δtau = 2, Δi ∈ {-1,0}), so **f = 1, c = 0 suffices** —
fdtd needs no new schedule math, only the two matcher/emitter
generalizations: (a) per-phase bands (already engine input per 4.1);
(b) the 1-D border phase as a degenerate 2-D phase with i-band
[0,1) — the window clipping then orders it before phase 1's rows
automatically.  Note the skew slack is P·Tt = 4·Tt per dim (twice
jacobi's), so the auto-Ts derivation matters more here; expect a
smaller optimal Tt (sweep {8, 16, 24}).

Acceptance: fdtd-2d XL ≥ 1.2x, SINK bit-identical, jacobi/heat/seidel
unchanged.

## 5. WP4 — symm (fission + scalar promotion + interchange + EXISTING peel)

v3 skipped symm; the v3 in-place peel changed the calculus, because
symm's hard kernel is *one interchange away* from a shape we already
crush. Kernel:

```
for i { for j {
    temp2 = 0                                  // scalar
    for k < i { C[k][j] += alpha*B[i][j]*A[i][k];   // scatter (k varies)
                temp2  += B[k][j]*A[i][k]; }        // reduction
    C[i][j] = beta*C[i][j] + alpha*B[i][j]*A[i][i] + alpha*temp2;
} }
```

Verified IR (cgeist -O0, XL, /tmp/fuseinv/symm/base.mlir): `temp2` is
ALREADY a rank-0 `memref<f64>` alloca with plain affine load/store —
the scalar-promotion pass speculated in early drafts is unnecessary and
is dropped.  The j-body is: `store 0 → temp2; for k < i { C[k][j] +=
alpha·B[i][j]·A[i][k];  temp2 += B[k][j]·A[i][k] }; epilogue store
C[i][j]`.

Two pieces, plus one honest negative result:

1. **Statement fission of the k-loop** (existing
   dr-affine-loop-distribute machinery): scatter-part and temp2-part
   share NO written memref (scatter writes C; temp2-part writes only the
   rank-0 temp2) — fission of the k-loop is trivially legal.  Splitting
   the epilogue from the k-loops at depth j is the real dependence
   question (epilogue reads temp2 + C[i][j]; scatter writes C rows
   < i — disjoint at depth j): `checkMemrefAccessDependence` decides,
   do not hand-wave it.
2. **Triangular interchange in canonicalize**: the scatter nest
   `for i { for j { for k < i: C[k][j] += f(i,j,k) } }` has its
   accumulator varying in the INNER loops and invariant in `i` — after
   interchanging i innermost it is the distinct-accumulator triangular
   reduction the v3 peel + vectorizers already handle.
   `canonicalizeOnce` currently refuses non-constant reduction bounds
   (AffineRegisterBlock.cpp:488); replace that bail with a
   dependence-checked interchange (upstream
   `isValidLoopInterchangePermutation` is the gate, not bound shape).
3. **The temp2 half stays scalar — by analysis, not neglect.**  Its
   accumulator is rank-0: invisible to BOTH vectorizers (broadcast
   needs the acc load stride-1 in the lane dim; rank-0 has no dims) and
   to BOTH Stage-3 gates (each requires `addrDependsOnIV(store, sIn)`,
   AffineRegisterBlock.cpp:1682 and the 3b sweep).  Vectorizing it
   along k is also out: B[k][j] is k-strided (row stride 20.8 KB) →
   gather.  Since scatter and temp2 are each ~half the kernel's FLOPs,
   Amdahl caps symm at ~2x with the scatter half alone — consistent
   with Polly's own 1.50x ceiling here.  If the acceptance bar is
   missed, the documented fallback is a 1-element→j-vector temp2
   expansion (privatization), which is NEW mechanism and out of v4
   scope.

Then the existing machinery takes over: peel, jam, vectorize.

Acceptance: symm XL ≥ 1.3x (Polly 1.50x), SINK bit-identical, and —
because canonicalizeOnce got more permissive — the FULL dense set
re-validated (the interchange gate change is the riskiest line of this
WP; trmm/lu lit tests must still pin their behavior).

## 6. WP5 — gramschmidt: tile-granularity refusion (close 1.56x → ~2x)

Diagnosis from v3: the split projection sweep re-streams A and Q once
per nest per k; Stage 3b made the dot fast but the A-update nest still
streams column-major, and the two sweeps no longer share cache residency.
Polly fuses-and-tiles them.

The principled mechanism (NOT a gramschmidt hack): **block-interleave
mode for distribute** — when the locality guard finds shared streams
between units AND a downstream win in at least one unit, instead of
choosing split-vs-don't, emit:

```
for jj (block):                 // block sized so shared WS fits L2 (MachineModel)
  unit1 over j in [jj, jj+B)    // R-dot, vectorized by Stage 3b
  unit2 over j in [jj, jj+B)    // A-update
```

i.e. fission at *block* granularity: each unit still becomes a clean
perfect sub-band (matchers fire), but the shared `A[·][j-block]` /
`Q[·][k]` stay resident across the pair. This composes with WP-2's
"ask-don't-impersonate" principle: distribute decides B from
MachineModel, and decides *whether* via the same ReductionMatch dry-run
the regblock driver uses (`canCapture(unit)` exported from WP2's
ReductionMatch — the gramschmidt guard hole closes properly, replacing
the v3 optimism band-aid).

Scope guard: implement for the 2-unit shared-stream case only; covariance
/atax/bicg shapes must take their existing paths untouched (lit-pin the
distribute rationale output on all three).

Acceptance: gramschmidt XL ≥ 1.9x; atax/bicg/covariance/correlation
unchanged (these all flow through the same guard — full dense
re-validation mandatory).

## 7. WP6 — diagnosis-only items (timeboxed, one session total)

- **gemver** (campaign XL medians, total runtime 46 ms: cgeist-base
  45.97 ms, drcomp-fuse 39.4 (best ours), reg-block 41.4, clang 41.5,
  polly 35.1, and distribute-tile REGRESSES to 51.0): the signal is
  fusion-shaped — Polly beats even clang -O2, our fusion recovers part
  of the gap, and distribution actively hurts.  Diagnose what Polly
  fuses that drcomp-fuse refuses (emit-rationale on gemver's four
  nests); the distribute regression should also fall out of WP5's
  ask-don't-impersonate guard for free — verify.  46 ms total with
  3-run medians is noise-prone: 10+ iters for any conclusion.
- **lu** (1.00 vs 1.04): Polly's 4% is within two noise bands of the
  campaign; measure 10 iterations before believing it exists at all.
- **doitgen**: pre-existing pipeline FAIL, one root-cause session
  (tracked since v3 spec; still not a cost-model issue).

## 8. Sequencing, risk, and the single-config goal

Order: **WP1 → WP2 → WP3.1 (extraction) → WP3.2 (seidel) → WP4 (symm)
→ WP3.3 (fdtd) → WP5 (gramschmidt) → WP6.** Rationale: WP1/WP2 are
prerequisites for every later WP touching their files; seidel is the
largest single win; symm before fdtd because it reuses WP2's output
directly and its risk is concentrated in one gate change; WP5 last
among features because it perturbs the most-shared pass (distribute).

After WP3.3: introduce the composed config
`drcomp-v4 = distribute(+block-interleave) → stencil-time-tile →
dr-affine-loop-tile → register-block` and require it to be within 5% of
per-kernel best-ours on all 29 kernels — the "one compiler, not a config
sweep" deliverable.  Ordering constraints, both verified against current
behavior: (a) distribute may run before the stencil pass — it already
refuses to fission stencil t-bodies (the inter-nest t-carried dependence;
established in the v2 investigation) — but (b) the stencil pass MUST
precede dr-affine-loop-tile: the matcher requires perfect constant-bound
space nests, and any tiler strip-mining destroys the match (today the
tiler happens to reject the stencil bands via its reuse gate; do not
rely on a profitability gate for a structural precondition).  Matcher
disjointness: ping-pong/in-place stencils have no k-invariant
accumulator and reduction bands have no time loop; pin both directions
with lit tests (stencil pass no-op on gemm, regblock no-op on
jacobi-2d).

Checkpoint discipline: one checkpoint per WP (refactor WPs commit with
"no intended behavior change" + the validation evidence in the message).
The v2/v3 protections stay absolute: SINK gate after every change;
stop-the-line at < 0.97x vs `none` for any kernel that had ≥ 1.0x;
SMALL dump-diff for every new transform (seidel, fdtd, block-interleave)
BEFORE any XL timing.

## 9. Non-goals (unchanged from v3, plus)

- No ISL / general polyhedral scheduler — WP3's engine derives skews for
  the *matched* shapes only; if the dependence cone doesn't fit the
  skew-factor model, it bails to no-op.
- No diamond/hexagonal tiling, no OpenMP, no autotuning.
- No attempt at nussinov/floyd-warshall/durbin (ties; dynamic-programming
  shapes need algorithm-level work outside this compiler's thesis).
- WP2 does NOT redesign pass managers or invent a plugin system: four
  files, explicit structs, same pass entry point. Resist scope creep —
  the test for every abstraction added is "does seidel/fdtd/symm
  concretely consume it"; if not, it goes.
