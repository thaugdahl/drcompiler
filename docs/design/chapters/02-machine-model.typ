#import "../theme.typ": *

= The Machine Model <sec:mm>

`drcompiler::MachineModel` (`include/drcompiler/Analysis/MachineModel.h`) is
*one resolved description of the target machine*, shared by every cost-modelled
pass. Before it existed, each pass carried its own `l1`/`l2`/`l3`/latency/sharer
options with independent defaults — and they drifted: the design notes record a
stale L2 default that was 4× too small on one path, so two passes silently
costed *different machines*. The Machine Model makes that impossible: there is
one struct, one set of defaults, one JSON source, and one override rule.

== The resolution contract

Every field has three potential sources, in strict precedence. This is what
keeps a default run *bit-for-bit* identical to the pre-model compiler while
still letting a JSON describe a different machine.

#figure(
  cetz.canvas({
    import cetz.draw: *
    let bar(y, label, sub, kind) = {
      node((0, y), [#text(weight: 700)[#label]\ #text(size: 7pt, fill: pal.muted)[#sub]],
        kind: kind, w: 7.5cm, h: 1.05cm)
    }
    bar(0,    "3 · CLI option  — if explicitly set (hasValue())", "an override, not a source of truth · wins over all", "accent")
    bar(-1.4, "2 · cost-model JSON  (cpu-cost-model-file)", "overrides defaults · absent fields keep their default", "pass")
    bar(-2.8, "1 · built-in defaults", "the historical per-pass values · a no-JSON run is byte-identical", "mm")
    arrow((4.2, -2.5), (4.2, -0.3))
    content((5.6, -1.4), text(size: 7.5pt, fill: pal.accent, weight: 700)[precedence ↑])
    // gate note
    node((4.2, -4.4), [`hasExplicitVectorModel`, `hasExplicitThreadModel`,
      `hasExplicitGemmModel` — false until a JSON sets that group, so the new
      arms stay *inert* and the default machine is byte-identical.],
      kind: "io", w: 9cm)
    arrow((1.5, -3.3), (2.2, -4.0))
  }),
  caption: [The resolution contract. Defaults → JSON → explicit CLI. Three
  `hasExplicit*` latches keep every model arm added after v4 (vector, thread,
  GEMM) dormant unless a JSON opts in — so adding capability never perturbs the
  default.],
) <fig:contract>

== What the model holds, and who asks

The fields cluster into six groups. Each group answers a different family of
questions, and a different set of passes asks them.

#figure(
  cetz.canvas({
    import cetz.draw: *
    // center
    node((0, 0), [*MachineModel*\ #text(size: 7pt)[one machine,\ one description]],
      name: "mm", kind: "mm", w: 3.1cm, h: 1.7cm)
    // field groups (left)
    let fg(y, body, nm) = node((-6.2, y), body, name: nm, kind: "io", w: 3.6cm, h: 0.95cm)
    fg(3.0,  [Cache hierarchy\ #text(size:6.5pt)[l1/l2/l3, line, sharers, TLB]], "g1")
    fg(1.5,  [Vector model\ #text(size:6.5pt)[native/arch bits, regBudget]], "g2")
    fg(0.0,  [Roofline arms\ #text(size:6.5pt)[fmaUnits, BW/cycle]], "g3")
    fg(-1.5, [Thread model\ #text(size:6.5pt)[threads, SMT, exclusive]], "g4")
    fg(-3.0, [Registers + GEMM\ #text(size:6.5pt)[budgets, gemmBlocking]], "g5")
    for nm in ("g1","g2","g3","g4","g5") { arrow(nm + ".east", "mm.west") }
    // consumer passes (right)
    let cp(y, body, nm) = node((6.0, y), body, name: nm, kind: "pass", w: 3.4cm, h: 0.8cm)
    cp(2.4,  [`affine-register-block`], "p1")
    cp(1.0,  [`data-recomputation`], "p2")
    cp(-0.4, [`memory-fission`], "p3")
    cp(-1.8, [`affine-loop-tile`], "p4")
    cp(-3.2, [`stencil-time-tile`], "p5")
    for nm in ("p1","p2","p3","p4","p5") { arrow("mm.east", nm + ".west") }
    content((6.0, 3.4), text(size: 7.5pt, fill: pal.muted)[consumers (one machine each)])
    content((-6.2, 3.9), text(size: 7.5pt, fill: pal.muted)[field groups])
  }),
  caption: [The hub. Field groups compose one `MachineModel`; every cost-modelled
  pass consumes the same instance, so all decisions are made against one machine.],
) <fig:hub>

=== Cache hierarchy & contention
Sizes (`l1Size`/`l2Size`/`l3Size`, `cacheLine`) and latencies
(`l1Lat`…`memLat`), plus page/TLB reach (`pageSize`, `l2TlbEntries`,
`tlbReachBytes()`). Contention is first-class: the *effective* LLC a kept value
can count on is `effectiveLLC() = l3Size / llcSharers` — a reuse beyond it is
priced at memory latency, because a co-tenant can evict it. `effectiveCache(lv)`
additionally halves a private L1/L2 per SMT sibling. These feed the
recomputation and fission cost models (@sec:dr) and the GEMM tiler.

#callout("Relation to LLVM's own cache modelling", [
  LLVM is *not* cache-blind, and the Machine Model deliberately overlaps it — the
  point below is the *delta*, not a claim of novelty. Two upstream facilities
  already model memory:
  #list(
    [`TargetTransformInfo` exposes `getCacheSize`, `getCacheLineSize`,
      `getCacheAssociativity`, `getPrefetchDistance`, `getMinPageSize`. These are
      *sizes*, and only for two levels: the `CacheLevel` enum is `{L1D, L2D}`
      and its source comment states L3 is deliberately *not* modelled ("their
      sizes differ widely between microarchitectures … we currently do not have
      a use for L3 cache size modeling"). The base `getCacheSize` returns
      `std::nullopt`; a value appears only if the subtarget's scheduling model
      populates it, so on many x86 subtargets it is simply absent.],
    [`LoopCacheAnalysis` computes a per-loop *reuse* cost
      (`computeRefCost ≈ TripCount·stride / CacheLineSize`, classifying temporal
      vs. spatial reuse) and drives loop-interchange. Its cost is in *cache-line
      counts*, single-thread; it carries no per-level *latency* and no bandwidth
      term (verified: no `latency`/`bandwidth` in the header).],
  )
  So what the Machine Model adds over `TTI` + `LoopCacheAnalysis` is specifically:
  #list(
    [*L3 and LLC contention* — `l3Size`, `effectiveLLC() = l3Size / llcSharers`,
      the per-thread cache split — the level LLVM explicitly declines to model;],
    [*per-level latencies in cycles* (`l1Lat…memLat`), turning a line-count reuse
      estimate into a cycle-weighted one;],
    [*a bandwidth / roofline arm* (`streamCycles`, `ridgeIntensity`) — LLVM's
      `MCSchedModel` gives per-instruction throughput but no memory-bandwidth
      term, so it cannot classify bandwidth- vs. compute-bound;],
    [*a thread/SMT/exclusive contention model*, where LLVM's is single-core;],
    [*throughput-effective vs. architectural vector width*
      (`vectorBitsNative` ≠ `vectorBitsArch`) — the double-pump truth a
      per-instruction cost model misses;],
    [*the register file as a tiling budget* (`vecRegBudget`, `gemmBlocking`,
      `macroTile`) — a transform-driving decision made before regalloc, not a
      reaction to pressure after it.],
  )
  In one line: LLVM models cache *size and reuse* at two levels, single-thread;
  the Machine Model turns that into *latency- and bandwidth-weighted, L3- and
  contention-aware, transform-driving* budgets.
], fg: pal.mm, bg: pal.mmbg)

=== Vector-execution model
The throughput-optimal vector *element* count is the native FP datapath width in
elements, raised only if the accumulator tile would not fit the register file.
`preferredVectorElems(elemBytes, mr, nr)` encodes this (@sec:rb). The subtlety
the model captures: a vector *wider* than the native datapath buys *no* FLOPs —
on Zen4 a 512-bit FMA is two µops over the 2×256-bit pipes, the same rate as two
`ymm` FMAs — while halving architectural accumulator count (less ILP) and, on
Intel, tripping AVX-512 frequency licensing (`avx512FreqThrottle`). So
`vectorBitsNative` (256 on Zen4) governs throughput; `vectorBitsArch` (512) only
bounds the encodable width.

=== Roofline arms
Two arms, paired in @sec:roofline. The *bandwidth* arm `streamCycles(bytes)`
divides bytes by the per-thread share of `dram`/`llcBytesPerCycle`. The
*compute* arm `peakFlopsPerCycle(elemBytes) = 2 · lanes · fmaUnits` and
`computeCycles(flops)`. Their ratio is the ridge `ridgeIntensity()` that
classifies a kernel as bandwidth- or compute-bound — the decision that separates
a deep-K FFN GEMM from a tiny-K attention matmul. Both arms return `0` when
unmodelled, so the caller's latency estimate stands (byte-identical default).

=== Thread model
`activeThreads`, `smtPerCore`, per-level `BytesPerCycle`, and the
`INTERSPERSED` vs `EXCLUSIVE` deployment fork that decides whether a thread gets
`BW/activeThreads` or the full bandwidth. Defaults to a 1-thread no-op.

=== Registers & GEMM blocking
`RegisterBudget` (gp/fp/vec/pred) bridged from the JSON `registers` block — the
*full* file for register-pressure analysis, with `vecRegBudget` (24 of 32 `zmm`)
the *accumulator* slice the tiler may use. On top sit the configurator queries:
`gemmBlocking(M,N,K)` → a `GemmTiling{mr,nr,vl,kc,mc,nc,kind,cacheTile}`,
`canFitAccumulators`, `maxL1Kc`, and the pure `macroTile()` cache-tile shrinker.
These own the codegen decisions in @sec:rb.

#callout("Why this matters", [
  Each later chapter is a single arrow in @fig:hub made concrete. The roofline
  (@sec:roofline) is the *Roofline arms* group answering "bandwidth or compute?".
  Register blocking (@sec:rb) is `gemmBlocking` + `preferredVectorElems`
  answering "what tile?". Recomputation and fission (@sec:dr, @sec:fission) are
  the *Cache hierarchy* group answering "does keeping this value pay?". The case
  study (@sec:gap) is all of them answering one real question correctly.
], fg: pal.mm, bg: pal.mmbg)
