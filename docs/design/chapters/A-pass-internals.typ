#import "../theme.typ": *

#set heading(numbering: "A.1")
#counter(heading).update(0)

= Appendix — Pass Internals <sec:appendix>

The body chapters explain *why* each transform exists and how it ties back to the
Machine Model. This appendix is the companion: the *internal control flow* of each
pass as the code actually runs it — the matcher stages, the guard predicates that
make a rewrite refuse, and the rewrite mechanics. The figures here are flowcharts
of the source, in pipeline order:

#align(center, table(
  columns: (auto, auto, 1fr),
  inset: 5pt, align: (left, left, left),
  stroke: (x, y) => if y == 0 { (bottom: 0.8pt + pal.ink) } else { (bottom: 0.3pt + pal.gridln) },
  table.header([*§*], [*pass*], [*source*]),
  [A.1], [`raise-malloc-to-memref`], [`RaiseMallocToMemRef.cpp`],
  [A.2], [`data-recomputation`], [`DataRecomputation.cpp` + `Strategies/`],
  [A.3], [`memory-fission`], [`MemoryFission.cpp`],
  [A.4], [`dr-math-strength-reduce`], [`MathStrengthReduce.cpp`],
  [A.5], [`dr-scalar-reduction-demote`], [`ScalarReductionDemote.cpp`],
  [A.6], [`affine-register-block`], [`AffineRegisterBlock.cpp` + `RegisterBlock/`],
  [A.7], [`dr-scalar-reduction-promote`], [`ScalarReductionPromote.cpp`],
))

The codegen quartet (A.4–A.7) runs as one `func.func(...)` pipeline: strength-reduce
→ demote → register-block → promote. A.5 and A.7 are exact inverses bracketing the
vectorizer; read them together.

// ======================================================================== A.1
== `raise-malloc-to-memref` <app:raise>

The front of the pipeline. cgeist (Polygeist) emits heap arrays as
`call @malloc` returning `memref<?xi8>`, immediately bitcast to `!llvm.ptr` by
`polygeist.memref2pointer`, with all access through `llvm.getelementptr` +
`llvm.load`/`store`. This pass recovers typed `memref` ops so the affine machinery
downstream has something to analyze. `RaiseMallocToMemRefPass::runOnOperation`
runs six explicit phases; phases 1–3 are the malloc→memref core, 4–6 mop up
residual Polygeist casts.

#figure(
  cetz.canvas({
    import cetz.draw: *
    let r(y, body, nm, k) = node((-1.6, y), align(left, body), name: nm, kind: k, w: 8.4cm, h: 0.66cm)
    r(0.0,  [*recognize malloc* — `call @malloc`→`memref<?xi8>`, one `memref2pointer` user], "m", "io")
    r(-0.95,[*`analyzePtrUses`* — infer element type, classify every pointer use], "a", "pass")
    r(-1.9, [*`decomposeByteSize`* — extent from a constant or `muli(count, sizeof)`], "d", "pass")
    r(-2.85,[*`memref.alloc<N×T>`* + rewrite GEP / `load` / `store` → memref ops], "al", "win")
    r(-3.8, [*`free`→`memref.dealloc`*; erase `memref2pointer`, `malloc`], "fr", "win")
    arrow("m.south","a.north"); arrow("a.south","d.north")
    arrow("d.south","al.north"); arrow("al.south","fr.north")
    node((5.8, -0.95),
      [*bail*\ #text(size:6.5pt)[leave LLVM ops]\ #text(size:6pt, fill: pal.muted)[escape · type\ inconsistent · size\ ≠ `c`/`muli(n,sizeof)`]],
      name: "bail", kind: "loss", w: 3.0cm, h: 1.9cm)
    for a in ("m","a","d") {
      line(a + ".east", "bail.west", stroke: (paint: pal.loss, dash: "dashed"),
        mark: (end: ">", scale: .6, fill: pal.loss))
    }
  }),
  caption: [`raise-malloc-to-memref` phases 1–3. A malloc is recognised only if it
  returns a single `MemRefType` with exactly one `memref2pointer` user;
  `analyzePtrUses` infers the element type from the GEPs/stores and classifies
  every pointer use; `decomposeByteSize` recovers the extent from a constant byte
  count or a `muli(count, sizeof)`. Any escaping or inconsistent use sets
  `valid = false` and the LLVM ops are left for clang.],
) <fig:app-raise>

=== GEP-index → memref-index

The one piece of real index work is turning a pointer GEP back into a memref
subscript. cgeist's pattern is a single-index 1-D GEP, so the conversion is
direct — the *first dynamic GEP index* becomes the memref index; a direct access
to the base pointer is index `0`:

#figure(
  cetz.canvas({
    import cetz.draw: *
    node((-4.3, 0.0), [`basePtr`\ #text(size:6.2pt)[(`memref2pointer`)]], name: "bp", kind: "io", w: 2.2cm, h: 0.9cm)
    let u(y, body, k) = node((1.6, y), align(left, body), name: none, kind: k, w: 9.0cm, h: 0.64cm)
    u(1.5,  text(size:7pt, font: "DejaVu Sans Mono")[`gep[%i] → ld/st`   ⟹   `memref.load/store alloc[toIndex(%i)]`], "win")
    u(0.5,  text(size:7pt, font: "DejaVu Sans Mono")[`ld/st @ basePtr`    ⟹   `… alloc[0]`   (direct, no GEP)], "win")
    u(-0.5, text(size:7pt, font: "DejaVu Sans Mono")[`call @free`         ⟹   `memref.dealloc alloc`], "io")
    u(-1.5, text(size:7pt, font: "DejaVu Sans Mono")[`any other use`      ⟹   pointer escapes ⇒ `valid=false`], "loss")
    for y in (1.5, 0.5, -0.5, -1.5) { arrow("bp.east", (-2.9, y)) }
    content((1.6, -2.45), text(size: 6.8pt, fill: pal.mm)[index is literally `gepOp.getDynamicIndices()[0]`  ·  `toIndexType` unwraps an `index_cast`])
  }),
  caption: [Pointer-use classification + GEP lowering. The index is literally
  `getDynamicIndices()[0]` — no general stride linearization, because the input is
  always the 1-D cgeist shape. Phases 4–6 lower any leftover
  `pointer2memref`/`subindex` casts (to `memref.view` / `llvm.gep`) and otherwise
  leave them for the back end.],
) <fig:app-gep>

The pass is deliberately conservative: it raises only the exact malloc/GEP idiom
cgeist produces and bails to the LLVM form on anything else, so a partial match
never miscompiles — it just leaves work for the later `finalize-memref-to-llvm`.

// STUB-RAISE-END

// ======================================================================== A.2
== `data-recomputation` <app:dr>

The core analysis pass (~2900 LOC). It is an interprocedural abstract
interpretation that assigns every `memref.load` a *provenance* — the set of stores
that could have produced its value — and then, under `dr-recompute`, rewrites the
safely-replaceable ones. The driver runs in three movements: build module-wide
maps, run the per-function dataflow, then classify and transform.

=== The pipeline

#figure(
  cetz.canvas({
    import cetz.draw: *
    let s(p, body, nm, k, w: 11.0cm, h: 0.66cm) = node(p, body, name: nm, kind: k, w: w, h: h)
    s((0, 0),   align(left, [*resolve config* — `CpuCostModel` + `MachineModel::fromJson` → cache/arch/register params (CLI > JSON > handler default)]), "cfg", "mm")
    s((0, -0.95), align(left, [*build module maps* — alloc roots, `StoreValueDeps`, module global writers, call graph, phase analysis]), "maps", "io")
    s((0, -1.9),  align(left, [*per-function dataflow* — seed cross-fn writers, `analyzeBlock` abstract-interprets store/load/call/`if`/`for` → fills `loadProv`]), "df", "pass")
    s((0, -2.85), align(left, [*interproc load propagation* — per call edge, union the caller's per-arg reaching stores into callee loads]), "ip", "pass")
    s((0, -3.8),  align(left, [*classify* — each load's `StoreSet` → SINGLE / MULTI / LEAKED / KILLED (@fig:prov)]), "cl", "decision")
    s((0, -4.75), align(left, [*cost gate* (`dr-cost-model`) — group SINGLE loads by alloc root, `decideBufferStrategy` → `skipBuffers`]), "cg", "decision")
    s((0, -5.7),  align(left, [*strategy pipeline* (`dr-recompute`) — first-match rewrite per surviving SINGLE load (§ below)]), "st", "win")
    for a in ("cfg","maps","df","ip","cl","cg") {
      line(a + ".south", a + ".south", stroke: none)
    }
    arrow("cfg.south","maps.north"); arrow("maps.south","df.north")
    arrow("df.south","ip.north"); arrow("ip.south","cl.north")
    arrow("cl.south","cg.north"); arrow("cg.south","st.north")
  }),
  caption: [`data-recomputation` top level. Analysis (`loadProv`) is fully built
  before any rewrite, because the transform invalidates `Operation*` keys. The
  cost gate and the strategy pipeline are independently gated (`dr-cost-model`,
  `dr-recompute`), so the pass can classify-only, cost-only, or rewrite.],
) <fig:app-dr-pipe>

=== Classification — the exact predicate

A load's `StoreSet` (from `loadProv`) collapses to one of four classes by a
literal test (@fig:prov shows the lattice; here is the code):

#callout("StoreSet → class", [
  `contains(nullptr)` → *LEAKED*  ·  `empty()` → *KILLED*  ·
  `size()==1` → *SINGLE*  ·  else → *MULTI*.

  The `nullptr` sentinel (LEAKED) is injected two ways: `joinStoreMaps` adds it
  when a root is written on one control-flow path but not another (a
  may-not-be-written merge), and `applyCall`/`applyLLVMStore` add it for an
  external or flat-pointer write with no coverage. The empty set (KILLED) comes
  from `killDependentStores` or from concrete `PointSet` coverage where a later
  store fully overwrote the covered indices. Only *SINGLE* loads are rewrite
  candidates.
], fg: pal.mm, bg: pal.mmbg)

=== Strategies — first-match pipeline + the remat kernel

Each surviving SINGLE candidate `{loadOp, storeOp, storedValue}` is run through an
ordered, first-match-wins strategy pipeline (`LoadStrategy::tryApply` →
`Accepted` / `NotApplicable`). The split is structural — same-function vs
cross-function:

#figure(
  cetz.canvas({
    import cetz.draw: *
    node((0, 1.55), [SINGLE candidate\ `{loadOp, storeOp, storedValue}`], name: "cand", kind: "io", w: 5.6cm, h: 0.78cm)
    node((0, 0.5), [*`loadFn == storeFn` ?*], name: "q", kind: "decision", w: 4.2cm, h: 0.6cm)
    arrow("cand.south", "q.north")
    // same-fn column (left)
    let L(y, body, nm) = node((-3.2, y), align(left, body), name: nm, kind: "win", w: 5.7cm, h: 0.74cm)
    L(-0.85, [① *DirectForward*\ #text(size:6.3pt)[store dominates ⇒ forward]], "s1")
    L(-1.8,  [② *FullRemat*\ #text(size:6.3pt)[clone whole SSA tree]], "s2")
    L(-2.75, [②b *PartialRemat*\ #text(size:6.3pt)[clone leaf loads · budget+gate]], "s3")
    arrow("s1.south","s2.north"); arrow("s2.south","s3.north")
    // cross-fn column (right)
    let R(y, body, nm) = node((3.2, y), align(left, body), name: nm, kind: "accent", w: 5.7cm, h: 0.74cm)
    R(-0.85, [③ *InterprocRemat*\ #text(size:6.3pt)[map callee→caller · remat at call]], "s4")
    R(-1.8,  [④ *CrossFnOrdered*\ #text(size:6.3pt)[hoist to caller · thread arg]], "s5")
    arrow("s4.south","s5.north")
    // branch arrows from q
    arrow("q.south", "s1.north"); content((-2.7, 0.05), text(size:6.5pt, fill:pal.win, weight:700)[same fn])
    arrow("q.south", "s4.north"); content((2.7, 0.05), text(size:6.5pt, fill:pal.accent, weight:700)[cross fn])
    content((3.2, -2.45), text(size:6pt, fill:pal.muted)[(④ runs as a separate later loop)])
  }),
  caption: [The strategy pipeline. Same-function loads try forward → full remat →
  partial remat; cross-function loads try interprocedural remat, with
  Strategy 4 (cross-function ordered) run as a separate pass over loads whose
  writer lives in another function. `Accepted` stops the chain for that load.
  Strategy 0 (constant-global fold) runs once, before analysis.],
) <fig:app-dr-strat>

Strategies 2/2b/3 share one engine, `RematKernel`:
*`isRematerializable`* walks the stored value's operand tree (worklist, depth ≤ 8,
≤ 64 ops) — chaining *through* another load only if it too is SINGLE-provenance,
cloning a dead-end load as a "partial leaf" only under a safety check + leaf
budget, and rejecting any op with memory effects or any call. Then
*`rematerializeAt`* clones the collected ops (topologically) just before the load,
`replaceAllUsesWith`, and erases the load. DirectForward is the degenerate case
(no clone — the value already dominates).

=== The cost gate

When `dr-cost-model` is on, SINGLE loads are grouped per alloc root and
`decideBufferStrategy` (`CacheCostModel.cpp`) prices keep vs recompute through the
arch's `combineCosts(mem, reg, alu)`:

#callout("decideBufferStrategy (per buffer)", [
  ```
  memKeep      = N · estimateLoadLatency(bufferSize + footprint)
  memRecompute = N · (leafLoadCost + operandPenalty)
  recompute = combineCosts(memRec, regRec, N·alu)
             <= combineCosts(memKeep, regKeep, alu+1)
  ```
  `estimateLoadLatency` tiers the buffer to L1/L2/effective-LLC/mem latency;
  beyond the effective private cache the load is floored at `streamCycles`
  (bandwidth arm). Partial-remat has a stricter intra-procedural gate
  `reject = (alu + leaf ≥ loadLat)` and a stride-aware leaf cost
  `ceil(missLat · min(cacheLine, stride·elemBytes+1) / cacheLine)`.
], fg: pal.accent, bg: pal.accentbg)

Every latency, size, and sharer count in that box is a Machine-Model field — the
same ones the fission gate (@app:fission) reads. Recomputation removes a buffer;
fission creates one; both decisions are the cache hierarchy priced the same way.

// STUB-DR-END

// ======================================================================== A.3
== `memory-fission` <app:fission>

`MemoryFissionPass::runOnOperation` works per function. Despite the name it is not
loop-nest fission: it finds *several separate `affine.for` loops that each
recompute the same expensive subexpression from the same source array*, and
materializes that subexpression into one shared buffer (@sec:fission). Three
phases: collect chains, group duplicates, then a cost-gated rewrite.

=== Collect & group — `extractChains` then bucket

#figure(
  cetz.canvas({
    import cetz.draw: *
    let s(p, body, nm, k, w: 3.0cm, h: 1.0cm) = node(p, body, name: nm, kind: k, w: w, h: h)
    s((0, 0), [find *expensive op*\ #text(size:6.2pt)[`opCost ≥ minChainCost` (15)]], "e", "decision", w: 3.0cm)
    s((3.5, 0), [forward-extend\ #text(size:6.2pt)[one use, free,\ `≥ minConsumerCost` (10)]], "f", "pass", w: 2.9cm)
    s((7.0, 0), [back-trace to\ `sourceMemref`\ #text(size:6.2pt)[sum `opCost`]], "b", "pass", w: 2.7cm)
    s((10.2, 0), [`buildFingerprint`\ #text(size:6.2pt)[SSA-name-free hash]], "fp", "io", w: 2.7cm)
    arrow("e.east","f.west"); arrow("f.east","b.west"); arrow("b.east","fp.west")
    // group
    s((5.2, -2.0), [group by\ (`sourceMemref`, fingerprint, parent region)], "g", "pass", w: 4.2cm, h: 0.95cm)
    darrow("fp.south", "g.east")
    s((10.2, -2.0), [candidate\ #text(size:6.2pt)[iff ≥ 2 loops]], "c", "decision", w: 2.7cm)
    arrow("g.east","c.west")
    content((5.2, -3.0), text(size: 6.8pt, fill: pal.muted)[`computeCost = max` over the grouped chains · `elementType = tip type`])
  }),
  caption: [Chain extraction. A `CompChain` runs from an `affine.load` (the
  `sourceMemref`) through a chain of memory-effect-free ops to an expensive "tip"
  (e.g. `sqrt → divf`). The structural `fingerprint` abstracts over SSA names so
  two loops computing the same expression hash identically; a `(memref,
  fingerprint, region)` group with ≥ 2 members is a fission candidate.],
) <fig:app-fission-collect>

=== The cost decision — two branches

The chapter (@fig:fission) shows the *fallback* predicate. The code actually has
two branches, selected on whether the head loop has a static trip count and a
scalar element type:

#grid(columns: (1fr, 1fr), gutter: 10pt,
  callout("Primary — contention-aware (static, scalar)", [
    ```
    recomputeC  = N·computeCost + N·srcBW
    materializeC = computeCost + 1 + N·bufLat
    materializeWins = materializeC < recomputeC
    sourceThrashes  = perConsumerFP > effL3
    fission = materializeWins || sourceThrashes
    ```
    `bufLat = max(tierLat, streamCycles)`; `srcBW`/`streamCycles` are the roofline
    bandwidth arm (0 without a thread model). `effL3 = l3Size / llcSharers`.
  ], fg: pal.mm, bg: pal.mmbg),
  callout("Fallback — the @sec:fission form (dynamic / non-scalar)", [
    ```
    recomputeCost = N·computeCost
    keepCost = computeCost + 1 + N·l1Latency
    fission = keepCost < recomputeCost
    ```
    `load_lat` optimistically assumed `l1Latency`; this is the simplified
    predicate the concept chapter quotes.
  ], fg: pal.muted, bg: white),
)

The primary branch adds two things the fallback omits: the `srcBW` term prices
*re-reading the source* N times (not just the recompute ALU), and the
`sourceThrashes` clause forces fission whenever recompute would re-stream the
source past the effective LLC — a capacity argument independent of the roofline
arithmetic. Both pull from the same Machine-Model cache fields as the
recomputation gate (@app:dr), which is the point: one machine, mirror decisions.

=== Guards & rewrite

After the decision, three guards must hold or the candidate is dropped: the
source memref must be *read-only* across all consumers (no stale buffer), no
consumer body may contain a *nested loop* (complex bodies unsupported), and all
loops must share *identical bounds*. The rewrite then allocates a 1-D
`memref<?×T>`, emits one producer `affine.for` that clones the chain and stores
its tip to the buffer, rewrites every consumer's tip to an `affine.load` of the
buffer, and `dealloc`s after the last consumer.

// STUB-FISSION-END

// ======================================================================== A.4
#pagebreak(weak: true)
== `dr-math-strength-reduce` <app:math>

A two-phase function pass (`MathStrengthReduce.cpp`). Phase 1 is an exact
strength reduction of integer-power `powf`; phase 2 is an optional lowering of the
remaining transcendentals to polynomial arithmetic. Both exist to delete the
`call` barrier that pins a pointwise loop to the scalar unit (@sec:transc).

#figure(
  cetz.canvas({
    import cetz.draw: *
    let r(y, body, nm, k) = node((-1.5, y), align(left, body), name: nm, kind: k, w: 7.6cm, h: 0.66cm)
    r(0.0,  [*`fn.walk`* — collect every `math.powf`], "walk", "io")
    r(-0.95,[*const float exponent?*  (`m_ConstantFloat`)], "c1", "decision")
    r(-1.9, [*integer-valued & in `[1, maxExponent]`?*], "c2", "decision")
    r(-2.85,[*`powByMul`* → multiplies · `replaceAllUsesWith` · erase], "rw", "win")
    arrow("walk.south","c1.north"); arrow("c1.south","c2.north"); arrow("c2.south","rw.north")
    // skip branch to the right
    node((5.4, -1.42),
      [*no* ⇒ leave\ `math.powf`\ #text(size:6pt, fill: pal.muted)[non-integer /\ huge exponent]],
      name: "skip", kind: "loss", w: 2.8cm, h: 1.5cm)
    for n in ("c1","c2") {
      line(n + ".east", "skip.west", stroke: (paint: pal.loss, dash: "dashed"),
        mark: (end: ">", scale: .6, fill: pal.loss))
    }
    // phase 2 below
    node((0, -4.0),
      align(left, [*phase 2* (if `poly-approx`): `populateMathPolynomialApproximationPatterns` → greedy rewrite of `exp` / `tanh` / `erf` / … to call-free vector arith]),
      name: "p2", kind: "accent", w: 11.0cm, h: 0.85cm)
    darrow("rw.south", "p2.north")
  }),
  caption: [`dr-math-strength-reduce` control flow. Phase 1 walks every
  `math.powf`; only a *constant, integer-valued* exponent in `[1, maxExponent]`
  is rewritten, so the transform is exact (independent of fast-math) and anything
  else is left bit-identical. Phase 2 is the opt-in poly-approx lowering.],
) <fig:app-math>

*Phase 1 — `powf(x, n)` → multiplies.* `powByMul` is exponentiation by squaring,
so `n` costs `⌊log₂n⌋ + popcount(n) − 1` multiplies (n = 3 → 2). It walks the bits
of `n`, squaring a running `base` and folding it into `result` on each set bit:

#figure(
  cetz.canvas({
    import cetz.draw: *
    let cell(x, body, k) = node((x, 0), text(size: 7.5pt, font: "DejaVu Sans Mono")[#body], name: none, kind: k, w: 2.0cm, h: 0.7cm)
    content((-1.7, 0.95), text(size: 8pt, weight: 700)[`n = 3 = 0b11`])
    content((-1.7, 0), text(size: 7.5pt)[bit loop:])
    cell(0.7, [bit 0 = 1\ res = x], "io")
    cell(3.0, [sq: base = x·x], "pass")
    cell(5.3, [bit 1 = 1\ res = res·base], "win")
    cell(7.7, [x · x²\ → x³], "win")
    arrow((1.7, 0), (2.0, 0)); arrow((4.0, 0), (4.3, 0)); arrow((6.3, 0), (6.7, 0))
    content((4.0, -0.9), text(size: 7pt, fill: pal.muted)[2 `arith.mulf`, exact — works on scalar *and* `vector<vl>` operands alike])
  }),
  caption: [`powByMul` for n = 3: the classic square-and-multiply bit loop. Because
  it is integer exponentiation it is exact, so it fires with no fast-math
  precondition — unlike the back end, which leaves `math.powf` a `call powf@plt`.],
) <fig:app-pow>

Both phases are pure function-local rewrites; no Machine-Model query is involved
in *whether* to fire (it always does when the shape matches). The Machine Model
only sets the *width* the now-call-free loop later vectorizes to, via
`preferredVectorElems` in register-block (@sec:transc).

// ======================================================================== A.5
#pagebreak(weak: true)
== `dr-scalar-reduction-demote` <app:demote>

`ScalarReductionDemote.cpp` rewrites onnx-mlir's `iter_args` SSA reductions into
the *memref-accumulator* form the register-block matcher expects (@sec:rb). It
collects matches in a `walk`, then rewrites (mutating during the walk is unsafe —
the rewrite erases and creates loops).

=== The matcher — `matchReduction(red, m)`

A cascade of guards; *any* failure returns `false` and the reduction is left
exactly as it was (correct, just unvectorized):

#figure(
  cetz.canvas({
    import cetz.draw: *
    let g(y, body) = node((0, y), align(left, body), name: none, kind: "decision", w: 12.5cm, h: 0.72cm)
    g(0.0,  [*1 result, 1 init, f32/f64* — `red.getNumResults()==1 && getInits().size()==1`, elem `isF32||isF64`])
    g(-0.95,[*descend the band* — each level is a single-`iter_arg` loop threading the accumulator; innermost yields `addf(acc, product)`; `product ≠ acc`; depth ≤ 4 (GEMM = 1 level, 3×3 conv = ic/kh/kw = 3)])
    g(-1.9, [*single use of the band result* — consumed by a store *directly* (Case A) *or* by one `addf` epilogue whose result is stored (Case B: `band + bias`)])
    g(-2.85,[*Case B legality* — only if the seed is `+0.0` (additive identity) and the bias is clonable into the init nest (its operands dominate the band)])
    g(-3.8, [*accumulator address ⟂ every band IV* — a true reduction, not a scatter (`store.getMapOperands()` contains no reduction IV)])
    g(-4.75,[*enclosing perfect spatial band* — each enclosing `affine.for` holds only its inner loop; `red`'s block holds *only* `{band, store, (epilogue, bias-def)}`])
    for i in range(5) {
      let y = -0.45 - i*0.95
      line((0, y), (0, y - 0.04), stroke: pal.line, mark: (end: ">", scale: .6, fill: pal.line))
    }
    content((6.4, -5.45), text(size: 7pt, fill: pal.win, weight: 700)[all pass ⇒ record Match{red, store, init, addf, redBand, spatial, (epilogue, epiBiasDef)}])
  }),
  caption: [`matchReduction`: the guard cascade. Case A is plain `store(band)`;
  Case B folds an onnx-mlir conv `band + bias` epilogue by seeding the
  accumulator with the bias (sound only because the band seed is `0`). The
  "perfect spatial band" walk *stops* (does not bail) at the first imperfect
  enclosing loop — onnx-mlir's trip-1 group `affine.apply` between `oc` and `oh`
  is tolerated.],
) <fig:app-demote-match>

=== The rewrite — `emitInitNest` + `demoteReduction`

The fission is two new nests, replacing the one `iter_args` loop:

#figure(
  cetz.canvas({
    import cetz.draw: *
    // before
    node((0, 0), align(left, text(size: 6.6pt, font: "DejaVu Sans Mono")[
      `%r = affine.for k`\
      `   iter_args(%a=%c0){`\
      `  %p = mulf A[i,k] B[k,j]`\
      `  %s = addf %a, %p`\
      `  yield %s }`\
      `store %r, C[i,j]`
    ]), name: "b", kind: "loss", w: 4.2cm, h: 2.4cm)
    content((0, 1.6), text(size: 8pt, weight: 700, fill: pal.loss)[`iter_args` — no memref acc])
    node((5.0, 0), [demote], name: "t", kind: "accent", w: 1.6cm, h: 0.9cm)
    arrow("b.east", "t.west")
    // after: init nest + reduce nest
    node((9.6, 1.0), align(left, text(size: 6.6pt, font: "DejaVu Sans Mono")[
      `affine.for i,j {`\
      `  store %init, C[i,j] }`
    ]), name: "init", kind: "io", w: 4.6cm, h: 0.95cm)
    node((9.6, -0.9), align(left, text(size: 6.6pt, font: "DejaVu Sans Mono")[
      `affine.for i,j { for k {`\
      `  %c = load C[i,j]`\
      `  %s = addf %c, A·B`\
      `  store %s, C[i,j] }}`
    ]), name: "red", kind: "win", w: 4.6cm, h: 1.4cm)
    arrow("t.east", "init.west"); arrow("t.east", "red.west")
    content((9.6, 1.85), text(size: 7.5pt, weight: 700, fill: pal.io)[init nest (own spatial loops)])
    content((9.6, -1.85), text(size: 7.5pt, weight: 700, fill: pal.win)[perfect reduction band])
  }),
  caption: [`emitInitNest` emits a *separate* spatial nest that seeds `C[i,j]`
  (Case B clones the bias def here); `demoteReduction` rebuilds the band with *no*
  `iter_args`, doing `load`/`add`/`store` of the accumulator in the innermost
  body. The init store lives in its own nest so the reduction band stays *perfect*
  — a leading store inside the band body would fail the vectorizer's perfect-body
  check.],
) <fig:app-demote-rw>

// ======================================================================== A.6
#pagebreak(weak: true)
== `affine-register-block` <app:rb>

@sec:rb gives the conceptual pipeline and the micro-kernel; this is the internal
stage machine. `AffineRegisterBlockPass::runOnOperation` runs one long fixed
sequence per `func.func`. Most stages are *shape canonicalizers* that massage a
nest into the one form the vectorizer matches; only the last few emit vectors. The
two cost-model gates (`hasExplicitVectorModel` for `vl`, `hasExplicitGemmModel`
for alloca-GEMM + cache tiling) keep the default byte-identical.

#figure(
  cetz.canvas({
    import cetz.draw: *
    let g(t) = text(size: 6.4pt, fill: pal.dec)[ · gate: #t]
    let row(y, tag, body, k) = node((0, y),
      align(left, [#text(size:7pt, weight:700)[#tag] #h(3pt) #text(size:7pt)[#body]]),
      name: none, kind: k, w: 11.6cm, h: 0.62cm)
    row(0.0,  [0], [derive `vl = preferredVectorElems` over FP accumulators#g[vector model]], "mm")
    row(-0.8, [0b], [`raiseSymmScatter` — split symmetric reduction + scatter], "pass")
    row(-1.6, [1], [`canonicalizeOnce`✻ — interchange reduction to innermost (i-k-j → i-j-k)], "pass")
    row(-2.4, [1.5], [`interchangeBlas2RowMajor` — col-major → stride-1 under sequential sweep], "pass")
    row(-3.2, [1.6], [*`canonicalizeAllocaGemm`* — scalar-alloca → 3 perfect nests (@sec:alloca)#g[gemm model]], "accent")
    row(-4.0, [1a], [family pre-scan — any `Dot` band ⇒ `reassoc`, shrink square tile], "decision")
    row(-4.8, [1c], [triangular peels✻ — diagonal / trmm / in-place (rect HEAD + scalar DIAG)], "pass")
    row(-5.6, [1b], [cache block — `gemmBlocking` (mc,nc,kc, L2) else legacy `macroTile`#g[`cacheTile`∨gemm]], "mm")
    row(-6.4, [1d], [`vectorizeConvBand` — multi-loop conv bands (compose+split+peel)], "win")
    row(-7.2, [2], [collect GEMM spatial loops — parallel `(sOut,sIn)` with acc-pair], "decision")
    row(-8.0, [3], [per band: family select → unroll-jam `mr` → *`vectorizeBroadcastBand`* / `vectorizeDotBand` / promote], "win")
    row(-8.8, [3b], [broadcast under sequential ancestor (projection sweeps)], "win")
    row(-9.6, [4], [set `fastmath` — `contract` (broadcast) / `fast` (dot)], "io")
    for i in range(12) {
      let y = -0.31 - i*0.8
      line((0, y), (0, y - 0.18), stroke: pal.line, mark: (end: ">", scale: .55, fill: pal.line))
    }
    content((0, -10.15), text(size: 6.2pt, fill: pal.muted)[✻ = run to fixpoint (`while changed`)])
  }),
  caption: [The `affine-register-block` driver. Stages 0–1.6 canonicalize; 1a–1b
  pick family and cache tiles from the Machine Model; 1d–4 emit vectors. A nest
  that fails to reach a matchable form at any stage simply falls through to the
  scalar `promoteReductions` fallback in Stage 3 — never a miscompile, just a
  missed vectorization that A.7 then keeps in registers.],
) <fig:app-rb-driver>

=== Matcher vocabulary

Every stage is built from a few predicates over `affine.for`:

#callout("The matcher primitives", [
  - *`onlyChildFor(loop)`* — the *perfect-nest* test: returns the sole nested loop
    iff the body holds exactly one loop and no other op (the "Stage-1b refuses
    imperfect nests" of @sec:alloca). A stray op ⇒ `nullptr` ⇒ must first be made
    perfect by `distributeLoop`/`canonicalizeAllocaGemm`.
  - *`isInnermost(loop)`* — no directly-nested `affine.for`.
  - *`collectAccumulators(k)` → `Acc`* — same-memref/map/operands `load`/`store`
    pair where the address is *k-invariant* and the stored value *reduces* the
    loaded value; an aliasing read of the acc memref is rejected (the LU
    `A[i][k]*A[k][j]` guard) unless certified `kAccNoAliasAttr`.
  - *`detectFamily` → {Broadcast, Dot}* — over the multiplicand loads:
    a multiplicand stride-1 in `j` ⇒ `Broadcast` (wide independent lanes, no
    reassoc); stride-1 only in `k` ⇒ `Dot` (reduction over `k`, small square
    tile, needs reassoc).
], fg: pal.mm, bg: pal.mmbg)

`MachineModel::gemmBlocking` is consulted at exactly one site (Stage 1b): the pass
takes only `cacheTile` and `(mc,nc,kc)` from it — `mr`/`nr`/`vl` come from the
options plus the Stage-0 `vl` derivation. (`GemmTiling.kind` and the
`OuterProduct`/`Gemv` arms are reserved; the current skeleton always returns
`Broadcast`.)

=== Broadcast micro-kernel emission — `vectorizeBroadcastBand`

The kernel of @fig:ukernel is emitted in a fixed order, behind guards that bail to
scalar on anything unsafe:

#figure(
  cetz.canvas({
    import cetz.draw: *
    let s(y, body, k) = node((0, y), align(left, body), name: none, kind: k, w: 12.0cm, h: 0.6cm)
    s(0.0,  text(size:7pt)[*guards:* `vl ≥ 2`, `sIn.step==1`, j-body all memory-effect-free, every acc load stride-1 in `j`, `canVectorizeDAG`], "decision")
    s(-0.8, text(size:7pt)[*vl-tail peel* — clone a scalar tail for `trip % vl` (output columns disjoint ⇒ independent)], "pass")
    s(-1.6, text(size:7pt)[*set step* `sIn.setStep(vl)`; hoist acc index `affine.apply`s above the loop], "pass")
    s(-2.4, text(size:7pt)[*inits* — one `affine.vector_load` per accumulator → `iter_args`], "io")
    s(-3.2, text(size:7pt)[*vector k-loop* — `vectorizeReductionValue`: acc→iter_arg, B→`vector_load`, A→`broadcast`, ×/+→vector op], "win")
    s(-4.0, text(size:7pt)[*write-back* — one `affine.vector_store` per acc after the loop; erase the scalar reduction], "win")
    for i in range(5) {
      let y = -0.3 - i*0.8
      line((0, y), (0, y - 0.2), stroke: pal.line, mark: (end: ">", scale: .55, fill: pal.line))
    }
  }),
  caption: [`vectorizeBroadcastBand`. The `mr × ⌈nr/vl⌉` accumulator grid of
  @fig:ukernel is built across two unroll-jams — `mr` from the Stage-3
  `loopUnrollJamByFactor(sOut, mr)` *before* this runs, and `⌈nr/vl⌉` from a
  `loopUnrollJamByFactor(sIn, nrVec)` *after*. The vector FMA materialises once
  Stage 4 stamps `contract` fast-math. The j-body-all-pure guard is load-bearing:
  it is what stopped a covariance `mean[j]=0` miscompile.],
) <fig:app-rb-bcast>

=== `canonicalizeAllocaGemm` legality (verbatim)

@sec:alloca shows the rewrite (promote rank-0 `alloca` → `C[i,j]`, fission into
INIT/GEMM/EPI) and the guard *intent*. The guards as the code checks them, in
order — any failing `continue` leaves the scalar nest:

#table(columns: (auto, 1fr), inset: 5pt, stroke: (x,y) => (bottom: 0.3pt + pal.gridln),
  align: (left, left),
  [*guard*], [*predicate*],
  [perfect not-yet-band], [`nred==1 && kLoop && onlyChildFor(jLoop)!=kLoop`],
  [single rank-0 local alloca], [`kaccs.size()==1 && accTy.getRank()==0 && allocaOp->getParentOp()==jLoop`],
  [additive reduction only], [`isa<arith::AddFOp, arith::AddIOp>(redOp)`],
  [accumulator local], [all acc users satisfy `jLoop->isAncestor(u)`],
  [one ≥1-D output store, i/j-indexed], [`n==1` and every `cOps` value is `iLoop.IV` or `jLoop.IV`],
  [*`C` write-only*], [`jLoop.walk(AffineLoadOp ld){ if ld.memref==cMemref readsOutput=true }; if readsOutput continue`],
  [store derives from acc], [backward-walk `cStore.value` reaches an `AffineLoadOp` on acc],
  [clean INIT segment], [pre-`k` ops are only `{alloca, hoistable init store}`],
)

The single *write-only* guard is the one that kills four bug classes at once —
in-place `C==A`/`C==B`, `beta`-accumulation `C = acc + C`, and a k-loop that reads
`C` — each of which an adversarial review constructed and confirmed is left scalar.

// STUB-RB-END

// ======================================================================== A.7
#pagebreak(weak: true)
== `dr-scalar-reduction-promote` <app:promote>

`ScalarReductionPromote.cpp` is the exact inverse of A.5, run *after*
register-block. The demote pass put *every* onnx-mlir reduction into memref form;
register-block re-promotes the bands it actually vectorizes, but every band it
*declined* is left accumulating into a real heap output tensor — one DRAM
round-trip per innermost iteration. Promote lifts those leftovers back to
`iter_args` so a missed vectorization degrades to *registers*, not DRAM (measured
3.2 s → 2.1 s on whole-resnet50; the WP-O1 assumption that backend mem2reg cleans
a heap buffer up is false).

#figure(
  cetz.canvas({
    import cetz.draw: *
    // match column
    let g(y, body) = node((0, y), align(left, body), name: none, kind: "decision", w: 12.5cm, h: 0.72cm)
    g(0.0,  [*innermost loop, no results* — `rb::isInnermost(inner)`, `getNumResults()==0` (a loop already carrying results is partly SSA already — skip)])
    g(-0.95,[*≥1 memory accumulator* — `rb::collectAccumulators`: same-address `load`/`store` pairs whose address is invariant across the band])
    g(-1.9, [*body is clean* — every other op is a `load` or memory-effect-free (a stray store/call would be reordered illegally by the rebuild)])
    g(-2.85,[*grow band upward* — through perfectly-nesting parent loops whose IV *no* accumulator address depends on (`addrDependsOnIV`)])
    g(-3.8, [*addresses available above the band* — every accumulator operand is defined outside the outermost band loop])
    for i in range(4) {
      let y = -0.45 - i*0.95
      line((0, y), (0, y - 0.04), stroke: pal.line, mark: (end: ">", scale: .6, fill: pal.line))
    }
    content((6.4, -4.5), text(size: 7pt, fill: pal.win, weight: 700)[`promoteBand`: load each acc once before → rebuild band carrying N scalar `iter_args` → store each once after])
  }),
  caption: [`matchBand` then `promoteBand`. The N > 1 case is the register-block
  *mr-jam* leftover (WP-G2 safety net): a band that slipped vectorization keeps
  `mr` same-shape accumulators in its innermost body; promoting *all* of them is
  what keeps a missed vectorization in registers. The pass is generic over the
  reduction op (works for `addf` chains, `maxnumf`, fused mul-adds) because the
  alias guard makes deferring the store sound.],
) <fig:app-promote>

#callout("A.5 and A.7 are an exact inverse pair", [
  Demote takes `iter_args → memref`; promote takes `memref → iter_args`. Between
  them sits register-block, which *wants* the memref form to match its accumulator.
  The round-trip is lossless for bands the vectorizer takes (it re-promotes them
  to vector `iter_args` itself) and a safety net for the ones it doesn't (promote
  restores the scalar `iter_args` the demote temporarily removed). Neither pass
  touches the Machine Model — they are pure shape adapters around the matcher.
], fg: pal.mm, bg: pal.mmbg)
