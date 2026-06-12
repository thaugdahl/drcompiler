# ONNX_CODEGEN_SPEC — Making the v4 codegen fire on onnx-mlir output

Status: IN PROGRESS (2026-06-12). Successor campaign to COSTMODEL_V4_SPEC.md.
WP-O1 DONE (commit `onnx_codegen_1`): demote pass + per-band family fix; 1x1
conv kernel 9.4x, resnet50 demotes 33/93 reductions, 17 vectorize. See
ONNX_CODEGEN_FINDINGS.md.
Evidence base: resnet50-v2-7 + mnist affine IR census (2026-06-11 spike, see
`memory/onnx_codegen_finding.md` and §1 below). Same methodology as v3/v4:
spike-first, honest verdicts, one local checkpoint commit per WP
(`onnx_codegen_N`), NEVER push.

---

## 0. The verdict we are responding to

The v4 pipeline (distribute → stencil-time-tile → register-block → tile) is a
clean NO-GO on onnx-mlir output: 0 vector ops on mnist and resnet50, loop
counts unchanged. Three structural blockers were isolated:

1. **IR form.** onnx-mlir emits every reduction as an SSA
   `affine.for ... iter_args` chain. The accumulator never touches memory
   inside the loop; only the final value is stored. The register-block matcher
   (`collectAccumulators` / `findAccPair`) hunts for an in-loop memref
   load/store accumulator *pair* — the cgeist -O0 shape. No pair → no-op.
2. **Conv structure ≠ GEMM shape.** Conv is a 3-deep reduction (ic, kh, kw)
   with the spatial loops (oc, oh, ow) *above* it. The Stage-2 gate requires
   the loop directly enclosing the innermost reduction to index the
   accumulator; for conv that loop is `kh` — itself a reduction. Fails even on
   a hand-written memref-form conv (verified).
3. **Batch-1 matmul = GEMV.** FC layers are BLAS-2 at inference batch size,
   already row-major; neither register-block nor the WP5 interchange applies.

This spec turns each blocker into a work package, sized by where resnet50's
FLOPs actually are.

---

## 1. What the resnet50 IR actually looks like (census, 5647 lines)

Pipeline that produced it: `onnx-mlir --O2 --EmitMLIR` →
`onnx-mlir-opt --convert-krnl-to-affine`. Counts:

| thing | count | note |
|---|---|---|
| `affine.for` | 844 | |
| `iter_args` reductions | 93 | all f32 scalar chains |
| `memref.alloc` | 204 | fresh tensor per op, no reuse |
| ReLU nests (`arith.maxnumf`) | 51 | each a separate full-tensor pass |
| 3×3 conv weights | 12×256², 8×128², 6×64², 6×512² | 4-D nests, padded |
| 7×7 stem | 2× `64x3x7x7` | stride 2 |
| 1×1 convs | ~16 distinct 2-D weight shapes | **already GEMM form, see below** |

### 1.1 The pivotal discovery: 1×1 convs are already GEMMs

onnx-mlir itself reshapes every 1×1 conv: spatial H×W is collapsed via
`memref.reinterpret_cast` and the nest emitted is a textbook GEMM —

```mlir
affine.for %n = 0 to #map(%batch) {          // parametric batch, trip 1
  affine.for %oc = 0 to 1024 {               // M
    affine.for %p = 0 to 196 {               // N  (14*14 collapsed spatial)
      %r = affine.for %ic = 0 to 256 iter_args(%a = %cst0) -> (f32) {
        %w = affine.load %W[%oc, %ic]        // A[m,k]
        %x = affine.load %X[%n, %ic, %p]     // B[k,n], p stride-1
        ... mulf, addf, yield
      }
      affine.store %r, %Y[%n, %oc, %p]       // C[m,n]
```

Constant bounds, stride-1 N dimension, single reduction loop, spatial loop
directly enclosing it. This is **exactly the shape the v4 broadcast
micro-kernel was built for**. The *only* blocker is the iter_args accumulator
form (blocker 1). Blocker 2 does not apply to 1×1 convs at all.

GEMM sizes across the net: M = oc ∈ {64…2048}, K = ic ∈ {64…1024},
N = collapsed spatial ∈ {3136, 784, 196, 49}.

### 1.2 FLOP distribution (batch-1, estimates)

- 1×1 convs ≈ **45–50%** of conv FLOPs → unlocked by WP-O1 alone, reusing the
  existing GEMM path unchanged.
- 3×3 convs ≈ **45–50%** → need the new direct-conv family (WP-O2).
- 7×7 stem ≈ 3%, final FC < 1% → not worth dedicated machinery.

### 1.3 Other load-bearing IR facts

- **Padding is in the bounds, not the data.** 3×3 conv kh/kw loops run
  `max #map(%oh)` to `min #map(%oh)` — non-constant bounds at the borders,
  constant `0..3` in the interior. Any conv vectorizer needs an
  interior/border split first.
- **Trip-1 group loop.** Every conv has `affine.for %g = 0 to 1` between batch
  and oc, plus an `affine.apply (d0 + d1*64)` rebuilding global oc. Must be
  promoted away or it breaks band perfection.
- **Parametric batch loop** `0 to #map(%batch)` encloses everything. The
  Vectorize.cpp / Peel.cpp constant-bound gates are band-local (verified:
  checks are on sOut/sIn/red only), so this *should* be tolerated — but band
  detection and canonicalizeOnce must be audited (WP-O4.3).
- **Every op writes a fresh alloc** and eltwise ops (bias add, ReLU, residual
  add) are separate whole-tensor nests. At 56²×256 an activation is 3.2 MB —
  larger than L2 — so each eltwise nest is a guaranteed round-trip to L3/DRAM
  (WP-O3).
- ReLU/maxpool reductions also use iter_args with `arith.maxnumf` — the
  de-promotion pass must not choke on non-add reductions (handle or skip
  cleanly).

---

## 2. Work packages

Order: **O0 → O1 → O4.1 → O2 → O3 → O4 (rest) → O5.** O1 is the highest
value-per-line item in the campaign; O2 is the largest. Each transform WP has
a spike gate — measure before building.

### WP-O0: Harness + honest baselines (½ day)

Deliverable: `scripts/onnx-codegen-bench.sh` — reproducible end-to-end run.

1. Lowering recipe (exists in /tmp form, productize): `onnx-mlir --O2
   --EmitMLIR` → `--convert-krnl-to-affine` → **host dr-opt** →
   `--convert-krnl-to-llvm` → mlir-translate → clang -O2 → link small C
   driver calling `run_main_graph` (or RunONNXModel.py).
2. Timing: median-of-N inference latency, batch 1, ≥1.5 s total per
   measurement (the seidel lesson: no sub-100 ms timescales).
3. Correctness: vectorized reductions reassociate FP adds, so **bit-identical
   is not the criterion here** (unlike PolyBench SINK). Criterion: max
   rel-err ≤ 1e-4 on output logits + top-1 class agreement on a fixed input
   set.
4. **Two baselines**, both reported: (a) `none` = same pipeline without
   dr-opt — isolates our transform effect; (b) `onnx-mlir --O3 --EmitLib` —
   their optimized krnl.matmul/conv path. (b) is the honest bar; beating only
   (a) is a finding about naive lowering, not about onnx-mlir.

### WP-O1: Reduction de-promotion pre-pass (the keystone, ~1–2 days)

New pass `dr-scalar-reduction-demote` (lib/Transforms/, registered in
Passes.td), run before the v4 pipeline. Two rewrites:

1. **Trip-1 promotion**: run `promoteIfSingleIteration` over the function
   (kills the group loop; the `affine.apply (d0+d1*64)` then folds).
2. **iter_args → memref accumulator**: for each innermost-perfect
   `affine.for ... iter_args` chain of f32 add-reductions whose result feeds
   (possibly via an epilogue op like bias-add) an `affine.store`:
   allocate a 1-element `memref.alloca`, initialize with the iter_args init
   value before the loop, replace the chain with load/op/store inside the
   loop, load the final value after. Nested chains (ic→kh→kw yielding through
   each level) collapse to a single accumulator.

Design decision — **pre-pass, not matcher extension**. Alternative considered:
teach `Acc`/`findAccPair` a second provenance (`IterArgsAcc`). Rejected for
v1: it touches every stage of the register-block pass (Stage-2 gate, jam,
promotion, peels, both vectorizers), whereas the pre-pass is one ~150-line
file with its own lit tests, and register-block *re-promotes to iter_args
anyway* — the round-trip is exactly its normal output. Chains the pass leaves
behind in loops we never transform are cleaned up by mem2reg in the LLVM
backend (alloca of scalar → SSA), so the de-promotion is performance-neutral
where we don't fire. Revisit native IterArgsAcc only if the pre-pass
measurably hurts something.

Scope guards: only f32/f64 `addf` chains (skip `maxnumf` pooling/softmax
chains cleanly); only when every iter_args use is the yield chain; tag
created accumulators with an attr for debugging.

**Spike gate (before writing the pass):** hand-edit one de-promoted
`1024×256, N=196` 1×1 conv nest, run existing `drcomp-v4` register-block on
it. Expect: vector ops > 0 and a measured kernel speedup vs the iter_args
original. If register-block fires but is *slower* than what LLVM already does
with the scalar form, stop and reassess (risk R1).

Acceptance: lit tests (chain forms: single, nested 3-deep, maxnumf skipped,
epilogue bias-add, non-yield use bails); resnet50 1×1 conv layers vectorize
end-to-end; mnist still correct.

### WP-O2: Direct-conv register-block family (the big one, ~1 week)

New family in the register-block pass (alongside GEMM/BLAS-2/scatter), firing
on the post-WP-O1 memref form of K×K convs (K ∈ {3, 7}, stride 1, groups
already promoted away):

```
for oc, oh, ow:                 // spatial
  for ic, kh, kw:               // reduction, acc = out[oc][oh][ow]
    acc += in[ic][oh+kh][ow+kw] * w[oc][ic][kh][kw]
```

1. **Matcher generalization.** Extend the accumulator analysis to accept a
   reduction *band* (up to depth 3) between the accumulator-indexing spatial
   loop and the innermost body: the Stage-2 gate becomes "store address
   depends on each spatial loop and on *no* loop in the reduction band"
   (today it checks only the single directly-enclosing loop — exactly why
   conv fails).
2. **Interior/border split (prerequisite).** kh/kw carry max/min padding
   bounds. Split oh and ow into [0,pad) ∪ [pad, U-pad) ∪ [U-pad, U) strips
   (pad=1 for 3×3); interior strips get constant `0..K` kh/kw bounds via map
   simplification; borders stay scalar. Reuse the Peel.cpp helpers/style.
3. **Micro-kernel** (conv-shaped sibling of the broadcast kernel):
   vectorize `ow` (stride-1 in both `out[..][ow]` and `in[..][ow+kw]` —
   unaligned vector loads on the input, fine), broadcast the scalar
   `w[oc][ic][kh][kw]`, unroll-jam `oc` by mr and `ow` by nr vector lanes,
   mr×(nr/vl) vector accumulators promoted to iter_args. This is what
   BLAS/oneDNN direct conv does; it cracks the same LLVM-won't-2D-vectorize
   wall the GEMM kernel cracked.
4. **Stride-2 convs: gated off in v1** (the downsample 1×1s and the stem).
   `in[.., 2*oh+kh, 2*ow+kw]` makes the vector load stride-2 (needs
   gather/deinterleave). The stride-2 1×1s still go through WP-O1+GEMM path
   only if their collapsed form is stride-1 — audit; otherwise they stay
   scalar. Log coverage lost.

**Spike gate (before any matcher work):** hand-write the ow-vectorized
micro-kernel in MLIR vector dialect for one `64×64×3×3 @ 56²` layer; time
against the scalar interior. Require **≥2× on the layer** to proceed. If the
spike fails, fall back to evaluating im2col (§4) before abandoning.

Acceptance: interior of every stride-1 3×3 conv in resnet50 vectorizes;
per-layer speedup ≥2× on the 4 weight classes; end-to-end resnet50 win
reported vs both baselines; no PolyBench regression (full
register-block lit suite + 6-kernel spot-check stays within 5%).

### WP-O3: Eltwise epilogue fusion (~2–3 days)

51 ReLU nests + bias/residual adds each stream a >L2 tensor through memory.
New pass `dr-eltwise-fuse` (or a mode of memory-fission — it is exactly the
inverse direction, reuse its cost-model plumbing):

- Fuse a purely elementwise consumer nest into its producer's store site when
  (a) iteration spaces are identical permutations, (b) the intermediate has a
  single consumer nest, (c) intermediate bytes > fusion threshold from
  MachineModel (`effectiveLLC()`-based — below it, the tensor is
  cache-resident and fusion buys little).
- Kills the intermediate alloc too (204 allocs today).
- Run *after* register-block: the eltwise op lands in the vectorized
  epilogue (apply to the yielded vector before the store) or in the scalar
  store for non-vectorized nests.

Spike gate: count DRAM-bound nests (tensor > L2) and estimate traffic saved;
hand-fuse one conv→ReLU pair at 56²×256 and measure before generalizing.

### WP-O4: Cost-model extensions (MachineModel/CpuCostModel)

1. **Small-N GEMM selection.** N ∈ {49, 196} breaks the mr=8/nr=16/vl=16
   default (N=49 → 3 vl-16 tiles + tail-17). Extend the family/config select:
   choose vl ∈ {16, 8, 4} and nr to minimize tail waste given N (the
   vl-tail-peel machinery exists; this is the *selection* logic). Drive from
   MachineModel vector width, not hardcoded.
2. **Conv tile selection (feeds WP-O2).** Weight slab for one (oc-block) of a
   512×512×3×3 conv is 512×9×4×mr bytes; full weights are 9.4 MB > L2. Add an
   ic-tiling decision: choose Tic so the live weight slab
   (mr×Tic×K²×4) fits L1 and the input row window fits L2 — formulas on
   MachineModel (`l1_size`, `l2_size`, `tlbReachBytes()` for the large-K
   weight walks).
3. **Parametric-outer-loop tolerance audit.** The batch loop is
   `0 to #map(%n)`. Band-local bound gates are already fine (verified);
   audit `canonicalizeOnce`, band detection, and the loop-tile footprint
   model so a parametric *enclosing* loop never bails a constant inner band,
   and footprint is computed per batch iteration.
4. **Fusion threshold** for WP-O3: `bytes(intermediate) >
   effectiveLLC()` (contention-aware — reuse the v3 sharers model).

### WP-O5: Composed config + evaluation (~2 days)

Pipeline (extends drcomp-v4; order matters, same reasons as v4 WP6):

```
inline, raise-malloc-to-memref,
func.func(dr-scalar-reduction-demote,        // NEW: trip-1 + de-promote
          dr-affine-loop-distribute,
          dr-affine-stencil-time-tile,
          affine-register-block{...},        // now with conv family
          dr-eltwise-fuse,                   // NEW: after RB
          dr-affine-loop-tile),
dr-pin-liveout
```

Evaluation matrix:
- mnist: sanity (expect ≈no change — it's maxpool+GEMV+softmax; the win is
  *not regressing* it).
- resnet50 batch-1: per-layer-class table (1×1 GEMM × {N=3136,784,196,49},
  3×3 × {64,128,256,512}ch) + end-to-end latency vs baseline (a) and (b).
- PolyBench no-regression: full lit + the v4 6-kernel spot-check within 5%.

---

## 3. Risks

- **R1 (top risk): the scalar baseline isn't naive.** LLVM -O2 loop-vec may
  already vectorize the de-promoted (or even iter_args) ic-reduction as a
  horizontal-sum dot. For 1×1 GEMMs the inner-product form has stride-196 B
  accesses, so our N-stride-1 broadcast kernel should win big — but *measure
  at the WP-O1 spike*, don't assume.
- **R2: onnx-mlir --O3 is a strong bar.** Their krnl.matmul is tiled+
  vectorized. Beating --O2-naive but losing to --O3 is a publishable
  comparison point, not a win — report both honestly.
- **R3: FP reassociation.** Vectorized reductions change summation order;
  define the tolerance criterion (WP-O0) *before* the first transform lands,
  so correctness goalposts never move.
- **R4: dynamic batch dimension** surprises in places not audited by O4.3.
  Mitigation: mnist+resnet50 both have it; it's exercised from day one.
- **R5: matcher generalization regresses PolyBench** (the v4 trmm lesson —
  every new gate needs the no-regression sweep). Mitigation: conv family is
  additive (new family select), full lit + spot-check per WP.

## 4. Non-goals

- **im2col conv→GEMM**: materializes the input ×K² (9× for 3×3) — pure added
  memory traffic in a regime where WP-O3 exists *because* traffic dominates.
  Only revisit if the WP-O2 spike fails its 2× gate.
- Maxpool/softmax vectorization (negligible FLOPs), the 7×7 stem and
  stride-2 convs (v1), groups>1 (ResNeXt), batch>1, quantized models,
  dynamic spatial shapes, alloc pooling (onnx-mlir's bundle-memory-pools owns
  that), GPU.

## 5. Mechanics (carried over from v4 — they bite)

- Host `ninja -C build dr-opt` edit loop; `/usr/bin/lit -s build/test/`.
- dr-opt runs host-side on staged .mlir; docker (`onnx-mlir-lean:x86_64`)
  only for onnx-mlir/lowering/run steps. Multiline/docker commands go in
  `/tmp/*.sh` scripts (lean-ctx mangles compound shell).
- `Option::hasValue()` not `getNumOccurrences()` for pipeline-set options.
- One commit per WP: `onnx_codegen_0` … `onnx_codegen_5`. NEVER push.
- Staged IR: `/tmp/onnx_spike/{mnist,resnet50-v2-7}.affine.mlir` (regenerate
  via the §WP-O0 script; /tmp is ephemeral).
