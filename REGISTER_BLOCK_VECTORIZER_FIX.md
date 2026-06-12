# REGISTER_BLOCK_VECTORIZER_FIX — making the direct-conv kernel fire on real resnet50 3x3

Status: SPEC (2026-06-12). This is **WP-O2 part 3b** of ONNX_CODEGEN_SPEC.md.
Prereqs landed: nested-band demote (`onnx_codegen_2a`), the direct-conv band
vectorizer `vectorizeConvBand` (`2b`), bias-epilogue folding / Case B (`2c`), and
the regression fix that made register-block safe on the de-promoted convs (`2d`).
Evidence base: `/tmp/onnx_spike/{conv3x3_*,conv_padded,conv_group}.mlir` and the
resnet50-v2-7 affine IR. Same methodology as the rest of the campaign:
spike-first, honest verdicts, one local commit per step, NEVER push.

---

## 0. Where we are

`dr-scalar-reduction-demote` + `affine-register-block` today:

- **1x1 convs** (the collapsed-spatial GEMMs) vectorize on real resnet50 — 136
  `vector.broadcast`, ≈ half the conv FLOPs. This is the WP-O1 win and it is
  intact.
- **3x3 convs** all de-promote to memref-accumulator bands (`iter_args` 93 → 0)
  but **do NOT vectorize**. `vectorizeConvBand` is proven (11.4x on a clean
  interior 3x3, `conv3x3_itargs.mlir`) yet bails on every real resnet50 3x3.

This spec closes that gap. The win is real but **partial by construction** (see
§5 coverage): only stride-1 3x3 convs with spatial extent ≥ VL benefit.

## 0a. Non-negotiable: register-block is fragile — test on real resnet50

`2d` fixed two bugs that **lit did not catch** because there is no whole-resnet50
lit case. Both came from feeding register-block IR shapes it had never seen.
Every change in this WP MUST be validated by:

```
dr-opt resnet50-v2-7.affine.mlir -allow-unregistered-dialect \
  --pass-pipeline='builtin.module(func.func(dr-scalar-reduction-demote,affine-register-block{mr=8 nr=16 vl=16}))' -o /dev/null
```

exiting 0 (valid IR) AND still emitting the 136 1x1 broadcasts. A green lit run
is necessary but NOT sufficient. (Generate the affine IR with `onnx-mlir --O2
--EmitMLIR` → `onnx-mlir-opt --convert-krnl-to-affine`, in the
`onnx-mlir-lean:x86_64` image; dr-opt runs host-side with
`-allow-unregistered-dialect` for the leftover `krnl.global` weights.)

---

## 1. The three blockers (with IR evidence)

The demoted real 3x3 res-block conv looks like this (constants folded for
clarity; `oc'` = the group-rebuilt output channel `affine.apply (g*64 + oc)`):

```mlir
// zero/bias init nest (already correct, Case B):
affine.for %oh = 0 to 56 { affine.for %ow = 0 to 56 {
  %b = affine.load %bias[%oc'] ; affine.store %b, %Y[%n, %oc', %oh, %ow] }}
// reduction band:
affine.for %oh = 0 to 56 { affine.for %ow = 0 to 56 {
  affine.for %ic = 0 to 64 {
    affine.for %kh = max(-%oh+1,0) to min(-%oh+57,3) {       // PADDED, dep. oh
      affine.for %kw = max(-%ow+1,0) to min(-%ow+57,3) {     // PADDED, dep. ow
        %a = affine.apply (d0,d1)->(d0+d1-1) (%kh, %oh)      // = oh+kh-1
        %b = affine.apply (d0,d1)->(d0+d1-1) (%kw, %ow)      // = ow+kw-1
        %i = affine.load %in[%n, %ic, %a, %b]                // index via APPLY
        %w = affine.load %wt[%oc', %ic, %kh, %kw]
        %c = affine.load %Y[%n, %oc', %oh, %ow]
        affine.store %c + %i*%w, %Y[%n, %oc', %oh, %ow] }}}}}
```

### B1 — load indices are hidden behind `affine.apply`

onnx-mlir precomputes every spatial index: `%i = load %in[.., %b]` where `%b =
apply(kw, ow)`. `ow` is **not a direct operand** of the load, so the ow-stride
analysis (`innermostStrideOne`, `canVectorizeDAG`) cannot see it. This is the
exact bug that mis-broadcast the stride-2 stem in `2d`; `vectorizeConvBand`
currently bails conservatively when a load reaches `ow` through an apply but is
not provably stride-1. **Every** real conv hits this, so today none vectorize.
(The hand spike used direct exprs `in[%ic, %oh+%kh, %ow+%kw]`, which is why it
worked.)

### B2 — the inner reduction bounds depend on the vectorized IV

`kw`'s bounds are `max(-ow+1,0) .. min(-ow+57,3)` — a function of `ow`. The 16
ow-lanes of one vector would each need a *different* `kw` trip count, which is
not expressible. Only the **interior** ow-range, where the clamp is inactive and
`kw ∈ [0,3)` is constant, can be vectorized. (`kh` depends on `oh`, not `ow` —
it is identical across all ow-lanes, so it does **not** block; leave it alone.)

### B3 — spatial extent is not a multiple of VL

Output widths are `ow ∈ {112, 56, 28, 14, 7}`. After the interior split the
vectorizable interior is `~[1, 55)` (width 54), `[1, 27)`, etc. None are
VL(16)-divisible, and `14`/`7` are **below** VL entirely. `vectorizeConvBand`
currently requires `trip(ow) % VL == 0` and bails otherwise — it needs the
vl-remainder peel that `vectorizeBroadcastBand` already has.

---

## 2. The fix, in three steps

Order: **C1 → C2 → C3.** Each is independently testable; C1 unblocks the
analysis, C2 creates the vectorizable region, C3 covers the tail.

### C1 — compose `affine.apply` into the band's load maps

Before analysing/transforming a conv band, normalise each `affine.load` /
`affine.store` in the band body so producer `affine.apply`s are folded into the
op's own affine map (`ow` becomes a real dim of the map). Use
`mlir::affine::fullyComposeAffineMapAndOperands` on `(map, operands)` and rebuild
the op, OR run the affine load/store canonicalization patterns over the band.

After composition:
- `in[.., ow+kw-1]` has innermost result `d_ow + d_kw - 1` → `innermostStrideOne`
  (already generalised to "coeff of ow == 1") returns true → contiguous vector
  load. ✓
- the **stride-2 stem** composes to `d_ow*2 + d_kw - 3` → coeff 2 → correctly
  NOT stride-1 → `vectorizeConvBand` still bails. ✓ (B1's safety guard becomes a
  precise check instead of a blanket bail.)

Scope: compose ONLY the loads/stores inside the band being vectorized (do not
canonicalize the whole function — that risks perturbing the 1x1 path that
already works). Remove the conservative `reachesOw`-through-apply bail from `2d`
once composition makes the stride test exact (keep a bail for any residual
non-stride-1 ow-dependent load — e.g. dilation, stride>1).

**Spike gate:** apply composition to `conv_group.mlir` (has the apply-ed
indices + group loop) and confirm `vectorizeConvBand` now fires on it
(constant-bound variant) with a correct checksum.

### C2 — interior/border split of `ow`

For a conv band whose innermost-but-one (`kw`) bounds depend on `ow`, split the
`ow` loop into three:

```
[0, owLo)        left border  — scalar, original clamped kw bounds
[owLo, owHi)     interior     — kw bounds REPLACED with the constant [cLb, cUb)
[owHi, W)        right border — scalar, original clamped kw bounds
```

Computing `[owLo, owHi)` from the bound maps (general, not hard-coded to pad=1):
the `kw` lb map is `max(eLb(ow), cLb)` and ub map is `min(eUb(ow), cUb)` where
`cLb`/`cUb` are the constant results and `eLb`/`eUb` the ow-dependent results.
The interior is where the constant dominates:
- `owLo` = least `ow` with `eLb(ow) ≤ cLb`  (e.g. `-ow+1 ≤ 0` → `ow ≥ 1`).
- `owHi` = `1 +` greatest `ow` with `eUb(ow) ≥ cUb` (e.g. `-ow+57 ≥ 3` → `ow ≤
  54` → `owHi = 55`).

Restrict v1 to the affine-linear case (coeff ±1 on `ow`, as all stride-1 convs
are): read the two results of each max/min map, identify the constant one, solve
the linear inequality for the other. Bail (leave scalar) on anything else.

Mechanics: clone the `ow` loop three times with the new constant lower/upper
bounds; in the interior clone, rewrite `kw`'s lower/upper bound maps to the
constant `cLb`/`cUb` (drop the ow-dependent result). The borders keep the
original band verbatim. The interior clone is then a clean constant-`kw` band
that C1+`vectorizeConvBand` can take.

Notes / risks:
- **Correctness pivot:** the interior must be *exactly* the region where the
  clamp is provably inactive, or border elements get wrong `kw` ranges. Verify
  with a SINK/checksum diff against the un-split scalar conv on a padded spike
  (`conv_padded.mlir`) BEFORE wiring vectorization.
- The `in` vector load in the interior reads `in[.., ow+kw-1]` for `ow ∈
  [owLo,owHi)`, `kw ∈ [0,3)`; the padded input halo guarantees in-bounds — but
  assert the vector main-loop upper bound (`owHi` rounded down to VL, see C3)
  keeps `ow+kw-1+VL-1` within the input extent.
- Do the split as a NEW stage in register-block (or a helper invoked from the
  Stage 1d conv detection) that runs BEFORE `vectorizeConvBand`. Keep it
  conv-only (guard on a multi-loop band with an ow-dependent inner bound) so the
  GEMM / PolyBench paths never see it.

### C3 — vl-remainder `ow`-peel in `vectorizeConvBand`

Generalise `vectorizeConvBand`'s `trip % VL == 0` requirement to peel the
remainder, mirroring `vectorizeBroadcastBand` (Vectorize.cpp:204–248): split the
(interior) `ow` into a VL-divisible main loop (vectorised) + a scalar tail clone
`[mainUb, owHi)`. The conv accumulator is a *disjoint output column per ow*
(like the broadcast kernel, NOT the dot kernel), so the tail is an independent
scalar copy — no ordering constraint.

`mainUb = owLo + ((owHi - owLo) floordiv VL) * VL`. If `owHi - owLo < VL` (the
14² and 7² layers), the main loop is empty — leave the whole interior scalar and
`log()` the coverage lost (or, later, retry with VL=8/4; out of v1 scope).

---

## 3. Putting it together (control flow in register-block)

New, conv-only, runs at Stage 1d (before the GEMM Stage 2/3), per detected conv
band `(sp=ow, band=[ic,kh,kw])`:

```
1. compose affine.apply into the band's load/store maps                 (C1)
2. if kw bounds depend on ow:                                           (C2)
     split ow into [borders | interior]; set interior kw := [cLb,cUb)
     recurse step 3 on the interior band only
   else: interior := the band as-is
3. vectorizeConvBand(interior_ow, band, VL)  -- now with the peel       (C3)
```

The borders and any sub-VL interior stay scalar (correct, just not accelerated).

## 4. Risks (ranked)

- **R1 — perturbing the working 1x1 path.** Everything here is gated to
  multi-loop conv bands; the GEMM stages must be byte-identical. Guard:
  resnet50 1x1 broadcast count stays 136; PolyBench register-block lit
  byte-identical. (This is the `2d` failure mode — treat it as the primary
  gate.)
- **R2 — border-split correctness.** An off-by-one in `[owLo,owHi)` silently
  corrupts edge pixels. Guard: checksum-diff the split-but-scalar conv vs the
  original scalar conv before any vectorization; max rel-err 0 (pure index-set
  split, no FP reorder).
- **R3 — FP reassociation in the vectorized band.** Per-lane order is preserved
  but `fastmath<fast>` (set when any band classifies Dot) may reassociate;
  acceptable under the onnx 1e-4 criterion, but report it. (The interior 3x3
  spike already measured 1e-6.)
- **R4 — input out-of-bounds on the vector tail.** The VL-wide load near `owHi`
  must stay within the padded input. Assert at build time.
- **R5 — stride-2 / dilation convs.** C1 makes these correctly NON-stride-1 →
  bail. Confirm the stem and the downsample convs stay scalar, not miscompiled.

## 5. Coverage (be honest about the payoff)

Per the resnet50 3x3 inventory and VL=16 (AVX-512 f32):

| spatial | layers (approx) | interior width | vectorizes? |
|---|---|---|---|
| 56×56 | 64-ch, early | ~54 | yes (3 VL + tail) |
| 28×28 | 128-ch | ~26 | yes (1 VL + tail) |
| 14×14 | 256-ch | ~12 | **no at VL=16** (< VL) |
| 7×7   | 512-ch | ~5  | **no at VL=16** |

So C1–C3 accelerate the 56² and 28² 3x3 convs (the early/mid blocks); the 14²
and 7² blocks (a large share of the *parameters*, but each spatial tile is tiny)
stay scalar until a smaller-VL path exists. Combined with the already-working
1x1s, this should cover the majority of resnet50's conv FLOPs — but the
end-to-end number is what decides it (WP-O0). **Do not claim a model-level
speedup from per-layer kernel wins.**

## 6. Validation plan

1. **C1 spike:** `conv_group.mlir` (apply-ed + group) constant-bound variant →
   `vectorizeConvBand` fires, checksum-identical to scalar.
2. **C2 spike:** `conv_padded.mlir` → split-but-scalar checksum == original
   scalar (rel-err 0); then split + vectorize → within 1e-4.
3. **C3:** a constant-bound conv with `ow = 54` (non-divisible) vectorizes with a
   correct tail.
4. **lit:** new tests for the interior/border split and the peel; full suite
   green; PolyBench register-block byte-identical; the `conv-gemm-coexist` guard
   still passes.
5. **Real resnet50 (the real gate, §0a):** valid IR, 1x1 still 136 broadcasts,
   and the 56²/28² 3x3 bands now show `vector.broadcast` + `affine.vector_store`.
6. **WP-O0 hand-off:** once 1x1 + 3x3 both vectorize, build the end-to-end
   harness and measure inference latency vs (a) no-dr-opt and (b) onnx-mlir
   `--O3` — the honest bar.

## 7. Non-goals (this WP)

Sub-VL spatial via VL=8/4 (the 14²/7² blocks), stride-2 / dilated convs, the 7x7
stem, depthwise/grouped convs, batch > 1, full-function affine canonicalization,
trip-1 group-loop promotion (already tolerated by the demote's early-stop, not
needed here).

## 8. Mechanics (carry-overs that bite)

- Host `ninja -C build dr-opt`; `/usr/bin/lit -s build/test/`.
- Lowering for spikes: marco LLVM 22 (`/home/tor/Dev/marco/install/llvm-project/bin`)
  — `mlir-opt` lower pipeline → `mlir-translate --mlir-to-llvmir` → `clang -O2`.
- Time at ≥1.5 s/measurement with an anti-hoist driver dependency (a bare REPS
  loop over a pure kernel is hoisted to one call — `driver3x3.c` perturbs `in`
  from the prior `Y`). FLOP-count the kernel before trusting a GFLOP/s number.
- One commit per step: `onnx_codegen_2e` (C1), `_2f` (C2), `_2g` (C3). NEVER
  push.
