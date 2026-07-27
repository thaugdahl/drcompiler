#import "../theme.typ": *

= Data Recomputation & Memory Fission <sec:dr>

The codegen family makes loops fast. The recomputation family asks the opposite
question — whether a memory access should exist at all. Both decisions price the
same cache hierarchy from the Machine Model.

== Load-store provenance

`data-recomputation` runs an interprocedural analysis that, for every
`memref.load`, computes the set of stores that could have written the value
(its *provenance*). Each load lands in one of four classes:

#figure(
  cetz.canvas({
    import cetz.draw: *
    node((0, 0), [`memref.load`], name: "ld", kind: "io", w: 2.0cm)
    let c(y, name, body, k) = node((5.2, y), body, name: name, kind: k, w: 6.4cm, h: 0.78cm)
    c(2.4,  "single", [*SINGLE* — exactly one known store · the recompute candidate], "win")
    c(0.9,  "multi",  [*MULTI* — several possible stores · ambiguous, keep], "plain")
    c(-0.6, "leaked", [*LEAKED* — provenance includes an unknown/external write · keep], "plain")
    c(-2.1, "killed", [*KILLED* — all reaching stores were killed · empty set], "plain")
    for nm in ("single","multi","leaked","killed") { arrow("ld.east", nm + ".west") }
    content((5.2, 3.2), text(size: 7.5pt, fill: pal.muted)[provenance classification])
  }),
  caption: [Load provenance. Only a *SINGLE*-provenance load can be safely
  replaced by rematerializing the one stored value at the load site; the others
  are kept. Classes and the analysis live in
  `Transforms/DataRecomputation/AnalysisState.h`.],
) <fig:prov>

When `dr-recompute` is set, a SINGLE load is rewritten to recompute the stored
expression in place — trading a memory access for the arithmetic that produced
it. That is only a win if the memory access was actually expensive, which is
where the cost model enters.

== The cache cost gate

`dr-cost-model` gates each rewrite on the latency arm of the Machine Model: the
load is priced at the latency of the smallest cache level that still holds the
value at its reuse distance, against the *effective* (contention-derated) LLC —
`effectiveLLC() = l3Size / llcSharers`. If the recomputed arithmetic is cheaper
than that priced access, the load is rematerialized; otherwise it stays.
`dr-partial-remat` extends this to clone a bounded number of non-chainable leaf
loads when the stride-aware leaf cost still clears the bar.

== Memory fission: the inverse of fusion <sec:fission>

Polyhedral fusion merges sibling loops to improve locality — but when several
siblings each *recompute* the same expensive subexpression, fusion has buried N
copies of it. `memory-fission` does the opposite: it *materializes* the shared
computation into a buffer once, and lets the consumers load it.

#figure(
  cetz.canvas({
    import cetz.draw: *
    // before: 3 siblings each recompute
    content((1.7, 2.4), text(size: 8pt, weight: 700, fill: pal.loss)[before — recompute ×N])
    for i in range(3) {
      node((1.7, 1.5 - i*0.85), text(size:7pt, font: "DejaVu Sans Mono")[`a#i[i]+=sqrt(x)/s …`],
        name: "s" + str(i), kind: "loss", w: 3.6cm, h: 0.6cm)
    }
    node((6.4, 0.65), [fission\ if keep < recompute], name: "t", kind: "accent", w: 2.2cm, h: 1.0cm)
    arrow((3.5, 0.65), "t.west")
    // after: producer + consumers
    content((10.6, 2.4), text(size: 8pt, weight: 700, fill: pal.win)[after — compute once])
    node((10.6, 1.5), text(size:7pt, font: "DejaVu Sans Mono")[`buf[i] = sqrt(x)/s`], name: "prod", kind: "win", w: 3.8cm, h: 0.6cm)
    for i in range(2) {
      node((10.6, 0.5 - i*0.7), text(size:7pt, font: "DejaVu Sans Mono")[`a#i[i] += buf[i] …`],
        name: "c" + str(i), kind: "io", w: 3.8cm, h: 0.55cm)
    }
    arrow("t.east", "prod.west")
    arrow("prod.south", "c0.north");
  }),
  caption: [Memory fission. Three siblings each recomputing `sqrt(x)/scale`
  (≈35 ALU cycles) become one producer that stores the value and consumers that
  load it (≈L1 latency). Fires per-candidate when
  `keep = compute + store + N·load_lat  <  recompute = N·compute`.],
) <fig:fission>

The break-even is a Machine-Model query: `compute` is the ALU cost of the
subexpression, `load_lat` is `l1Lat`/`l2Lat`/… for the level the buffer lands in,
and the buffer's residency is judged against the same effective-cache sizes. A
buffer that would spill past the effective LLC is priced at memory latency and
the fission is rejected — the same contention rule the recomputation gate uses.

#callout("One machine, two directions", [
  Recomputation removes a buffer (recompute instead of load); fission creates one
  (store once instead of recompute N times). They are mirror decisions over the
  *same* cost model — which is why both must read the *same* cache sizes and
  latencies. That shared truth is the Machine Model; the pre-model drift (a stale
  L2 default on one path) is exactly the bug it closed.
], fg: pal.mm, bg: pal.mmbg)
