#import "../theme.typ": *

= Cost Model & Roofline <sec:roofline>

The cost model answers one question for every candidate transform: _will moving
this work cost more in time than it saves?_ Time has two independent ceilings —
how fast the machine can *compute* and how fast it can *move bytes* — and the
Machine Model exposes both as the two arms of a roofline.

== The two arms

#grid(columns: (1fr, 1fr), gutter: 12pt,
  callout("Bandwidth arm", [
    `streamCycles(bytes, fromDRAM)` = `bytes / effBW`, where `effBW` is
    `dram`/`llcBytesPerCycle` divided by `activeThreads` (interspersed) or taken
    whole (exclusive). Returns `0` when bandwidth is unmodelled — the latency
    estimate then stands.
  ], fg: pal.io, bg: pal.iobg),
  callout("Compute arm", [
    `peakFlopsPerCycle(elemBytes)` = `2 · lanes · fmaUnits` with
    `lanes = vectorBitsNative / (8·elemBytes)`. `computeCycles(flops)` =
    `flops / peak`. Uses the *native* datapath width, not the arch width — a
    wider vector is double-pumped at the same FLOP rate. `0` when `fmaUnits=0`.
  ], fg: pal.pass, bg: pal.passbg),
)

The real move time of a tiled kernel is `max(latency, streamCycles, computeCycles)`
— whichever ceiling binds. The interesting question is *which one*, and that is
the ridge.

== The ridge: bandwidth- vs compute-bound

`ridgeIntensity(elemBytes, fromDRAM)` = `peak / effBW` is the arithmetic
intensity (flops per byte) at which the two arms cross. A kernel whose intensity
is *below* the ridge is bandwidth-bound — it is waiting on memory, and adding
compute blocking buys nothing. *Above* the ridge it is compute-bound, and
register blocking / tiling to keep operands resident is exactly the lever.

#figure(
  cetz.canvas({
    import cetz.draw: *
    let ox = 0.6
    let oy = 0.3
    line((ox, oy), (9, oy), mark: (end: ">", scale: .7), stroke: 1pt + pal.ink)
    line((ox, oy), (ox, 5), mark: (end: ">", scale: .7), stroke: 1pt + pal.ink)
    content((4.8, -0.3), text(size: 8pt)[arithmetic intensity  (flops / byte)  →])
    content((ox - 0.35, 2.7), trot(90deg, text(size: 8pt)[attainable flops/cycle →]))
    let rx = 4.6
    let ry = 3.9
    line((ox + 0.2, oy + 0.25), (rx, ry), stroke: 2pt + pal.io)
    line((rx, ry), (9, ry), stroke: 2pt + pal.pass)
    line((ox, ry), (rx, ry), stroke: (paint: pal.line, dash: "dotted"))
    line((rx, oy), (rx, ry), stroke: (paint: pal.line, dash: "dotted"))
    content((rx, ry + 0.3), text(size: 7.5pt, fill: pal.accent, weight: 700)[ridge = peak / BW])
    content((2.0, 2.55), trot(33deg, text(size: 7.5pt, fill: pal.io)[slope = bandwidth]))
    content((7.0, ry + 0.27), text(size: 7.5pt, fill: pal.pass)[peak = 2·lanes·fmaUnits])
    content((2.3, 0.7), text(size: 7.5pt, fill: pal.io, weight: 700)[bandwidth-bound])
    content((7.0, 1.2), text(size: 7.5pt, fill: pal.pass, weight: 700)[compute-bound])
    circle((2.2, 1.55), radius: 0.09, fill: pal.io, stroke: none)
    content((2.2, 2.0), text(size: 7.5pt, fill: pal.io)[QK#super[T] K=64\ (tiny-K)])
    circle((6.6, ry), radius: 0.09, fill: pal.pass, stroke: none)
    content((6.6, ry - 0.45), text(size: 7.5pt, fill: pal.pass)[FFN GEMM K=768\ (deep-K)])
  }),
  caption: [The roofline the Machine Model computes. A deep-K FFN GEMM sits on
  the compute ceiling (register-block + tile to keep it there); a tiny-K
  attention matmul sits on the bandwidth slope (more compute blocking is wasted).
  This is the classification `gemmBlocking` and the WP-T0 analysis used to pick
  *which* transform to spend on which kernel.],
) <fig:roofline>

#keyidea[
  The roofline is not decoration — it is the decision procedure. The openai-gpt
  case study (@sec:gap) found the FFN GEMMs compute-bound (so accumulator-chain
  ILP was the win) and the attention matmuls negligible — straight from this
  classification. Picking the wrong arm wastes the whole effort, as the
  bandwidth-bound gpt-neox attention showed.
]

== Cache pricing & contention

For the recomputation and fission passes the relevant arm is *latency* against a
contended cache. The model prices a reuse at the latency of the smallest level
that still holds it — but the level it can *count on* is derated:
`effectiveLLC() = l3Size / llcSharers`. A working set that fits L3 in isolation
but not `L3 / sharers` is priced at memory latency, because a co-tenant will
evict it. This single rule makes the keep-vs-recompute decision (@sec:dr)
contention-aware without every pass re-deriving it.

== Inert by default

Every arm added after the original latency model — the compute arm
(`fmaUnits=0`), the bandwidth arm (`BytesPerCycle=0`), the thread model — returns
`0`/no-op until a JSON sets it. So a run with no cost-model JSON computes the
*exact* costs it did before the roofline existed. Capability is opt-in; the
default is frozen.
