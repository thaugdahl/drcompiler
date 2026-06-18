// bubble-widening.typ — iterative region-formation algorithm, with figures.
// Companion to PARALLEL_BUBBLE_SPEC.md.  Native typst only (no external packages).
// Compile:  typst compile bubble-widening.typ

#set page(paper: "a4", margin: (x: 2.0cm, y: 2.2cm), numbering: "1")
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")
#set text(size: 10pt)
#show raw.where(block: true): set text(size: 8pt)
#show raw.where(block: true): set par(justify: false)

// ───────────────────────── palette ─────────────────────────
#let cA    = rgb("#1f77b4")  // bubble A  (init C)
#let cB    = rgb("#2ca02c")  // bubble B  (pointwise A->T)
#let cC    = rgb("#d62728")  // bubble C  (stencil T[i+1]->C)
#let cM    = rgb("#ff7f0e")  // merged region
#let cCall = rgb("#9467bd")  // consumed call
#let cCrit = rgb("#8c564b")  // peeled critical slab
#let cGray = luma(120)

// ───────────────────────── helpers ─────────────────────────
// A coloured "bubble" box wrapping a code region.
#let rb(label, col, code) = block(
  width: 100%, fill: col.lighten(88%), stroke: 0.9pt + col.darken(5%),
  radius: 4pt, inset: 7pt, above: 5pt, below: 5pt, breakable: false,
)[
  #text(size: 7.5pt, weight: "bold", fill: col.darken(30%))[#label]
  #v(-3pt)
  #code
]

// Plain (un-bubbled) context line(s).
#let ctx(code) = block(width: 100%, inset: (left: 7pt), above: 3pt, below: 3pt)[#code]

// A synchronization boundary between two regions.
#let barrier(txt) = align(center, block(
  width: 92%, fill: luma(232), stroke: (paint: cGray, dash: "dashed", thickness: 0.8pt),
  radius: 3pt, inset: 4pt, above: 5pt, below: 5pt,
)[#text(size: 8pt, weight: "bold", fill: luma(60))[#txt]])

// Narration of the widening move applied to reach the next figure.
#let move(txt) = block(width: 100%, inset: (left: 2pt), above: 4pt, below: 2pt)[
  #text(size: 9pt, fill: cGray)[#sym.arrow.b.double  *move:* #txt]
]

#let badge(col, txt) = box(
  fill: col.lighten(80%), stroke: 0.7pt + col, radius: 2pt, inset: (x: 4pt, y: 1pt),
)[#text(size: 7.5pt, fill: col.darken(30%))[#txt]]

// Worker→tile schematic: a 4×4 array of cells coloured by which worker owns each.
#let wcol = (rgb("#4e79a7"), rgb("#f28e2b"), rgb("#59a14f"), rgb("#e15759"))
#let cellbox(c) = box(width: 13pt, height: 13pt, fill: c.lighten(35%), stroke: 0.4pt + c.darken(10%))
#let tilegrid(fn) = {
  let cells = ()
  for r in range(4) { for c in range(4) { cells.push(cellbox(fn(r, c))) } }
  grid(columns: (13pt,) * 4, column-gutter: 1.5pt, row-gutter: 1.5pt, ..cells)
}

// ════════════════════════ title ════════════════════════
#align(center)[
  #text(size: 17pt, weight: "bold")[Bubble-Widening Region Formation]
  #v(-6pt)
  #text(size: 11pt, fill: cGray)[Iterative discovery of maximal parallel regions for drcompiler]
  #v(-2pt)
  #text(size: 9pt, fill: cGray)[companion to `PARALLEL_BUBBLE_SPEC.md` · figures · v1 2026-06-18]
]
#v(6pt)
#line(length: 100%, stroke: 0.6pt + cGray)

= Problem

drcompiler's cost model can reason about parallel *deployment* but, until the
sharding back-end, could not *emit* parallel code. The back-end
(`PARALLEL_CODEGEN_SPEC.md`) shards a single outermost parallel loop per hot nest.
This document specifies the *front-end* that decides what a parallel region even
*is*: rather than picking one loop, it grows *maximal* regions by an iterative
widening of per-loop *bubbles*, resolving conflicts by peeling, privatization, and
redistribution, and emits the runtime-agnostic `par` dialect for the back-end to
lower.

#block(width: 100%, fill: luma(245), stroke: 0.6pt + cGray, radius: 4pt, inset: 9pt)[
  #text(size: 9pt)[*Layering* (decision #5). This document is the top box; the
  existing sharding spec is the bottom box; the `par` dialect is the contract.]
  #v(3pt)
  #align(center)[
    #box(fill: cM.lighten(85%), stroke: 0.9pt + cM, radius: 4pt, inset: 7pt)[
      #text(size: 8.5pt, weight: "bold")[front-end (this spec)] \
      #text(size: 8pt)[`dr-par-bubbles`: seed → widen → peel/redistribute → maximal regions]
    ]
    #v(2pt) #text(fill: cGray)[#sym.arrow.b #h(4pt) emits `par.region` / `par.forall` / `par.{barrier,redistribute,critical,reduce}`]
    #v(2pt)
    #box(fill: cA.lighten(85%), stroke: 0.9pt + cA, radius: 4pt, inset: 7pt)[
      #text(size: 8.5pt, weight: "bold")[back-end (`PARALLEL_CODEGEN_SPEC.md`)] \
      #text(size: 8pt)[`par → libdrpar`: `decideShard()` cores/grain/domain · pinned pool · topology placement]
    ]
  ]
]

= The bubble abstraction

A *bubble* is an in-memory analysis object owning a single-entry/single-exit region
plus the loop axes it has proven parallel (`parDim`) versus sequential (`seqDim`).
Bubbles are seeded one per loop and *widen* by consuming frontier code:

#grid(columns: (1fr, 1fr), gutter: 8pt,
  [#badge(cM, "Climb") absorb the enclosing loop as a new parallel axis.],
  [#badge(cM, "Engulf") absorb an adjacent sibling op/region.],
  [#badge(cM, "Fuse") merge a conformant adjacent bubble.],
  [#badge(cC, "Freeze") stop at a barrier / unresolvable dependence.],
)

Legality is decided by one tiered predicate, `mayConflictUnderParallel` (Tier 0
allocation-root provenance · Tier 1 polyhedral affine dependence · Tier 2
conservative alias · cross-procedure forwarding). See spec §2.

= Algorithm

#block(width: 100%, fill: luma(247), stroke: 0.6pt + cGray, radius: 4pt, inset: 8pt)[
```text
seed:   one Bubble per affine.for/scf.for; classify each loop depth parDim | seqDim
widen:  worklist ← Active bubbles, innermost-first
        repeat
          changed ← false
          for B in worklist where Active:
            for move in {Climb, EngulfPred, EngulfSucc, Fuse}:
              verdict ← evaluate(B, move)             // ParAliasOracle + decideShard() gate
              Clean      → apply(B, move); changed ← true
              Resolvable → if costGate(fix): apply(B, move, fix); changed ← true   // privatize|peel|redistribute
                           else freezeEdge
              Hard       → freezeEdge;  if Fuse: recordRedistBoundary(B, other)
          until !changed                              // fixed point ⇒ maximal regions
emit:   each cost-positive bubble → par.region { par.forall(parDims) { … seqDims … } }
```
]

#pagebreak()

= Iterative walk-through

The running kernel: init `C`, a pointwise `A→T`, a stencil reading `T[i+1]` into
`C`, then a per-index `@scale(C)`. We watch the bubbles grow to a fixed point.

== Figure 1 — seed

Every loop becomes its own bubble; the call site is unclaimed context. Each loop's
axis `i` is independently parallel (`Tier 1`: no carried dependence within a loop).

#figure(
  block(width: 100%, stroke: 0.4pt + cGray, radius: 5pt, inset: 8pt)[
    #ctx[```mlir
func.func @stage(%A, %B, %C, %T : memref<Nxf32>) {
```]
    #rb([bubble A · par(i) · writes C], cA)[```mlir
affine.for %i = 0 to N { affine.store %z, %C[%i] }            // init C
```]
    #rb([bubble B · par(i) · reads A, writes T], cB)[```mlir
affine.for %i = 0 to N { affine.store math.exp(%A[%i]), %T[%i] }   // pointwise
```]
    #rb([bubble C · par(i) · reads T,C  writes C], cC)[```mlir
affine.for %i = 0 to N-1 { affine.store %C[%i] + %T[%i+1], %C[%i] }  // stencil
```]
    #ctx[```mlir
  func.call @scale(%C) : (memref<Nxf32>) -> ()     // per-index write, pure
}
```]
  ],
  caption: [Seed: three singleton bubbles A, B, C; the call is unclaimed.],
)

== Figure 2 — iteration 1: conformant fuse (A ⊕ B)

A writes `C`, B writes `T` — *disjoint allocation roots* (Tier 0), same trip `0..N`,
same worker mapping. The fuse is *conformant and clean* → A and B merge into one
region `M` distributing `i` once over both bodies.

#move[`Fuse(A, B)` — disjoint writes #badge(cA, "C") #sym.perp #badge(cB, "T"), conformant trips → #badge(cM, "CLEAN")]

#figure(
  block(width: 100%, stroke: 0.4pt + cGray, radius: 5pt, inset: 8pt)[
    #ctx[```mlir
func.func @stage(%A, %B, %C, %T : memref<Nxf32>) {
```]
    #rb([region M = A⊕B · par(i)], cM)[```mlir
affine.for %i = 0 to N {                       // one distributed i over both bodies
  affine.store %z, %C[%i]                      // (was A)
  affine.store math.exp(%A[%i]), %T[%i]        // (was B)
}
```]
    #rb([bubble C · par(i)], cC)[```mlir
affine.for %i = 0 to N-1 { affine.store %C[%i] + %T[%i+1], %C[%i] }
```]
    #ctx[```mlir
  func.call @scale(%C) : (memref<Nxf32>) -> ()
}
```]
  ],
  caption: [Iteration 1: A and B fuse (disjoint outputs, conformant). C unchanged.],
)

== Figure 3 — iteration 2: partial conflict → barrier boundary

C reads `T[i+1]`, produced by M (`T[i]`). Fusing M with C is *illegal*: iteration
`i` of C needs `T[i+1]` from iteration `i+1` of M (Tier 1: `Carried`, distance −1 on
`i`). Both regions are *individually* parallel, so we do *not* serialize — we
freeze the edge and keep two parallel regions separated by a synchronization
boundary (a `par.barrier`: all of `T` produced before any `T[i+1]` is read).

#move[`Fuse(M, C)` — offset-1 read of #badge(cB, "T") → #badge(cC, "HARD (carried −1)") → freeze, insert barrier]

#figure(
  block(width: 100%, stroke: 0.4pt + cGray, radius: 5pt, inset: 8pt)[
    #rb([region M · par(i)], cM)[```mlir
affine.for %i = 0 to N { affine.store %z,%C[%i]; affine.store math.exp(%A[%i]),%T[%i] }
```]
    #barrier[═══  par.barrier  (T fully produced before T[i+1] consumed)  ═══]
    #rb([region C · par(i)], cC)[```mlir
affine.for %i = 0 to N-1 { affine.store %C[%i] + %T[%i+1], %C[%i] }
```]
    #ctx[```mlir
  func.call @scale(%C)
```]
  ],
  caption: [Iteration 2: the offset-1 producer→consumer is a hard edge; both sides stay
  parallel, split by a barrier. (When mappings *differ*, this edge becomes a
  `par.redistribute` instead — §4.2.)],
)

== Figure 4 — iteration 3: consume the call (forwarding analysis)

C's following frontier is `@scale(%C)`. Call-site + forwarding analysis maps the
callee's effect to caller memref `C`: it writes `C[i]` per index, disjoint, pure
(no module-global, provenance not `LEAKED`). Verdict: *consumable* → engulfed into
C's region as `par.call`, same `i` mapping (no extra barrier).

#move[`EngulfSucc(C, @scale)` — callee writes #badge(cCall, "C[i]") per-index disjoint, pure → #badge(cCall, "CLEAN")]

#figure(
  block(width: 100%, stroke: 0.4pt + cGray, radius: 5pt, inset: 8pt)[
    #rb([region M · par(i)], cM)[```mlir
affine.for %i = 0 to N { affine.store %z,%C[%i]; affine.store math.exp(%A[%i]),%T[%i] }
```]
    #barrier[═══  par.barrier  ═══]
    #rb([region C⊕scale · par(i)], cCall)[```mlir
affine.for %i = 0 to N-1 { affine.store %C[%i] + %T[%i+1], %C[%i] }
par.call @scale(%C)                 // consumed: per-index disjoint write
```]
  ],
  caption: [Iteration 3: the pure per-index call is absorbed into the trailing region.],
)

== Figure 5 — fixed point: two maximal regions

No bubble grows in a further sweep. The kernel is two maximal parallel regions
joined by one barrier — far fewer fork/joins than sharding each of the four loops
independently.

#figure(
  block(width: 100%, stroke: 0.4pt + cM, radius: 5pt, inset: 8pt, fill: cM.lighten(95%))[
    #grid(columns: (1fr, auto, 1fr), align: horizon, gutter: 6pt,
      box(fill: cM.lighten(82%), stroke: 0.9pt + cM, radius: 4pt, inset: 7pt, width: 100%)[
        #text(size: 8pt, weight: "bold")[R1 = init C ⊕ pointwise A→T] \
        #text(size: 8pt, fill: cGray)[par(i), 0..N]
      ],
      align(center)[#text(size: 9pt, fill: cGray)[barrier #linebreak() #sym.arrow.r]],
      box(fill: cCall.lighten(82%), stroke: 0.9pt + cCall, radius: 4pt, inset: 7pt, width: 100%)[
        #text(size: 8pt, weight: "bold")[R2 = stencil ⊕ scale(C)] \
        #text(size: 8pt, fill: cGray)[par(i), 0..N-1]
      ],
    )
  ],
  caption: [Fixed point: two maximal regions, one barrier.],
)

== Figure 6 — materialized `par` dialect

Each region becomes a `par.region` with a `par.forall` over its parallel axis;
the back-end (`par → libdrpar`) then chooses cores/grain/domain via `decideShard()`.

#figure(
  block(width: 100%, stroke: 0.4pt + cGray, radius: 5pt, inset: 8pt)[
```mlir
par.region {                                   // R1
  par.forall (%i) in (0)..(N) {
    affine.store %z, %C[%i]
    affine.store math.exp(%A[%i]), %T[%i]
    par.yield
  } { mapping = #par.block }
}
par.barrier
par.region {                                   // R2
  par.forall (%i) in (0)..(N-1) {
    affine.store %C[%i] + %T[%i+1], %C[%i]
    par.call @scale_elem(%C, %i)
    par.yield
  } { mapping = #par.block }
}
```
  ],
  caption: [Materialization. `decideShard()` (back-end) sets cores/grain/domain per
  `par.region`; lowering to `omp.parallel` is the correctness oracle.],
)

#pagebreak()

= Fine-grained control

== Affine-overlap peeling

The Figure-3 edge was resolved by a barrier because the *whole* range conflicts at
distance 1. When a conflict covers only a *sub-polyhedron*, the conflicting slab
is *peeled* into a `par.critical` and the interior stays parallel — letting two
loops fuse for locality where a full barrier would otherwise be forced.

Example: producer `X[i]`, consumer `X[i+K]` with small constant `K`. The dependence
touches only the boundary; the interior `[K, N)` is conflict-free.

#figure(
  grid(columns: (1fr, auto, 1.1fr), align: horizon, gutter: 10pt,
    // before
    block(width: 100%, stroke: 0.4pt + cGray, radius: 5pt, inset: 7pt)[
      #text(size: 7.5pt, weight: "bold", fill: cGray)[before — full-range conflict] #v(-2pt)
```mlir
affine.for %i=0 to N { store X[%i] }   // prod
affine.for %i=0 to N { use  X[%i+K] }  // cons
```
      #v(2pt)
      #text(size: 7.5pt, fill: cC.darken(10%))[fuse ⇒ carried dep on X, distance K]
    ],
    align(center)[#text(size: 14pt, fill: cGray)[#sym.arrow.r]],
    // after
    block(width: 100%, stroke: 0.4pt + cCrit, radius: 5pt, inset: 7pt)[
      #text(size: 7.5pt, weight: "bold", fill: cCrit.darken(5%))[after — set-difference peel] #v(-2pt)
      #rb([interior K..N · par(i)], cM)[```mlir
par.forall (%i) in (K)..(N) {
  store X[%i]; use X[%i+K]    // fused, no carry
}
```]
      #rb([boundary slab 0..K · sequential], cCrit)[```mlir
par.critical { affine.for %i=0 to K { … } }
```]
    ],
  ),
  caption: [Peel: `FlatAffineValueConstraints` set-difference splits the domain
  (`PointSet::overlaps`/`-=` is the constant-index fast path). Gate: peel only when
  `interior ≫ slab`.],
)

== Redistribution — a transitory mapping change

A barrier (Figure 3) keeps the *same* worker→data map across the edge. When producer
and consumer disagree on that map — here a *row*-parallel producer feeding a
*column*-parallel consumer of the same `X` — a barrier alone leaves every consumer
worker reading rows produced by *every* producer worker (an implicit all-to-all). A
`par.redistribute` makes that reshuffle explicit and *costed*: it re-places `X`'s
residency from row-blocked to column-blocked across the pinned workers (the
`par → libdrpar` lowering realizes it with the back-end's `__drpar_alloc_local`
first-touch path).

#figure(
  block(width: 100%, stroke: 0.4pt + cGray, radius: 5pt, inset: 9pt)[
    #align(center)[
      #grid(columns: (auto, auto, auto), align: horizon, column-gutter: 16pt,
        stack(dir: ttb, spacing: 4pt,
          text(size: 7.5pt, weight: "bold", fill: cM.darken(20%))[producer · par(r)],
          tilegrid((r, c) => wcol.at(r)),
          text(size: 7pt, fill: cGray)[1 worker = 1 row band],
        ),
        stack(dir: ttb, spacing: 4pt,
          text(size: 17pt, fill: cCrit)[#sym.arrow.r],
          box(fill: cCrit.lighten(80%), stroke: 0.8pt + cCrit, radius: 3pt, inset: (x: 5pt, y: 3pt))[
            #align(center)[#text(size: 7.5pt, weight: "bold", fill: cCrit.darken(10%))[par.redistribute]]
          ],
        ),
        stack(dir: ttb, spacing: 4pt,
          text(size: 7.5pt, weight: "bold", fill: cCall.darken(10%))[consumer · par(c)],
          tilegrid((r, c) => wcol.at(c)),
          text(size: 7pt, fill: cGray)[1 worker = 1 column band],
        ),
      )
    ]
    #v(5pt)
    #block(width: 100%, inset: (x: 2pt))[
```mlir
par.region { par.forall (%r) in (0)..(M) { affine.for %c { X[%r,%c] = produce } } }   // row-parallel
par.redistribute %X : memref<MxNxf32>  from #par.block<dim=0> to #par.block<dim=1>
par.region { par.forall (%c) in (0)..(N) { affine.for %r { use X[%r,%c] } } }          // col-parallel
```
    ]
  ],
  caption: [Redistribution: the same `X`, re-blocked from rows (producer) to columns
  (consumer) across the four pinned workers. Cell colour = owning worker. Priced as
  `redistBytes / redistBytesPerCycle`, weighed against serializing one side.],
)

= Cost gate & hand-off

Every widening move and every materialization is gated by the back-end's
`decideShard()` (spec §7): a region is emitted only when `cores > 1`. With no
explicit `ThreadModel` the gate returns `cores == 1`, so *nothing is emitted* and
the output is byte-identical to today — the same default-OFF discipline as every
drcompiler pass. Two terms the front-end adds to `ThreadModel` (default 0):
`spawnCycles`, `barrierCycles` (multi-region widening creates many of both), and
`redistBytesPerCycle` for `par.redistribute`.

#align(center)[#text(size: 8pt, fill: cGray)[End of figures. Implementation milestones M0–M6 in `PARALLEL_BUBBLE_SPEC.md` §12.]]
