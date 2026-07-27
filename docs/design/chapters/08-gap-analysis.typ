#import "../theme.typ": *

= Case Study — Closing the openai-gpt Gap <sec:gap>

This chapter is the whole document at work on one question: _does our codegen
beat onnx-mlir's `--O3` on a real transformer (openai-gpt, batch 1, Zen4)?_ The
answer required separating two effects that a naïve comparison conflates, and
every fix came from a Machine-Model reading.

== Four configurations

To isolate *transform* quality from *back-end* quality, four builds of the same
model are timed, all correctness-checked (`norm-rel-err ≤ 1e-4` vs `none`):

#align(center, table(
  columns: (auto, auto, auto, auto),
  inset: 6pt, align: (left, left, left, right),
  stroke: (x, y) => if y == 0 { (bottom: 0.8pt + pal.ink) } else { (bottom: 0.3pt + pal.gridln) },
  table.header([*config*], [*transform*], [*back end*], [*median*]),
  [`none`], [onnx-mlir `--O2`], [host `clang`], [23.60 s],
  [*`codegen`*], [our `dr-opt`], [host `clang -O3`], [*0.231 s*],
  [`o3host`], [onnx-mlir `--O3`], [host `clang -O3`], [0.367 s],
  [`o3emitobj`], [onnx-mlir `--O3`], [onnx `--EmitObj`], [0.224 s],
))

`codegen` vs `o3host` share a back end, so their gap is *pure transform*.
`o3host` vs `o3emitobj` share a transform, so their gap is *pure back end*.

#figure(
  cetz.canvas({
    import cetz.draw: *
    let scale = 18.0 / 0.40   // cm per second-ish, schematic
    let bar(y, val, label, k) = {
      let w = val * scale * 0.5
      rect((0, y - 0.22), (w, y + 0.22), fill: (if k=="win" {pal.win} else if k=="loss" {pal.loss} else {pal.accent}), stroke: none)
      content((-0.15, y), text(size: 8pt)[#label], anchor: "east")
      content((w + 0.15, y), text(size: 8pt, weight: 700)[#val s], anchor: "west")
    }
    bar(3.0, 0.224, [o3emitobj], "accent")
    bar(2.0, 0.231, [*codegen*], "win")
    bar(1.0, 0.367, [o3host], "loss")
    content((4.2, 0.0), text(size: 7.5pt, fill: pal.muted)[(`none` = 23.60 s, off-scale — the scalar-alloca pathology)])
  }),
  caption: [The three optimized builds (lower is better). codegen beats `o3host`
  (same back end) by 1.59×; onnx-mlir's native `o3emitobj` edges codegen by 1.03×
  — a near-tie, within the measurement noise.],
) <fig:board>

== Two orthogonal gaps

#figure(
  cetz.canvas({
    import cetz.draw: *
    // axes: x = transform, y = backend
    line((0,0), (9,0), stroke: 0.8pt + pal.line)
    line((0,0), (0,5), stroke: 0.8pt + pal.line)
    content((4.5,-0.4), text(size:8pt)[transform quality  →])
    content((-0.35,2.5), trot(90deg, text(size:8pt)[back-end quality  →]))
    // four points
    let pt(x,y,lab,val,k) = {
      circle((x,y), radius: 0.12, fill: (if k=="win" {pal.win} else if k=="loss" {pal.loss} else {pal.accent}), stroke: none)
      content((x, y+0.45), text(size:7.5pt, weight:700)[#lab])
      content((x, y - 0.4), text(size:7pt, fill: pal.muted)[#val])
    }
    pt(2.2, 1.2, [o3host], [0.367 s], "loss")
    pt(7.0, 1.2, [*codegen*], [0.231 s], "win")
    pt(2.2, 4.0, [o3emitobj], [0.224 s], "accent")
    pt(7.0, 4.0, [(ours + native\ back end)], [≈0.18 s proj.], "plain")
    // gap arrows
    arrow((2.6, 1.2), (6.6, 1.2)); content((4.6, 1.55), text(size:7.5pt, fill: pal.win, weight:700)[transform gap 1.59×])
    arrow((2.2, 1.6), (2.2, 3.6)); content((3.5, 2.8), text(size:7.5pt, fill: pal.accent, weight:700)[back-end gap 1.66×])
    line((7.0,1.6),(7.0,3.6), stroke: (paint: pal.line, dash: "dashed"))
  }),
  caption: [The two gaps are orthogonal. Horizontally (same back end) our
  transforms beat `--O3` by 1.59×. Vertically (same transform) onnx-mlir's native
  `--EmitObj` beats host-`clang` by 1.66×. The product `o3emitobj` wins net only
  because the back-end edge slightly outweighs our transform edge — *not* because
  its transforms are better.],
) <fig:twogap>

== Every fix was a Machine-Model reading

#table(columns: (auto, 1fr), inset: 6pt, stroke: (x,y) => (bottom: 0.3pt + pal.gridln),
  align: (left, left),
  [*finding*], [*Machine-Model reasoning*],
  [GEMMs win the transform gap], [Roofline (@sec:roofline) classed the deep-K FFNs compute-bound; `gemmBlocking` gives 16 accumulator chains vs `--O3`'s 4, saturating the 2 FMA pipes (@sec:rb, @fig:ilp). 64% of matmul FLOPs.],
  [the "back-end gap" is transcendentals], [Disassembly: `o3emitobj` emits *0* `tanhf`/`expf`/`powf` (native vector polynomials), `o3host` 288/96/288 libm calls. Closed at the MLIR level by poly-approx (@sec:transc) — not a back-end mystery.],
  [vl=16 / single-`Kc`: no-go], [`preferredVectorElems` already picks vl=8 (native datapath) and `macroTile` already tiles to L2; forcing vl=16 or no-K-tiling is *slower* — the model was right (@sec:decisions).],
  [eltwise vectorizer: no-go], [The residual scalar arith was *not* the bottleneck — clang `-O3` auto-vectorizes the now-call-free poly loops better than an explicit fixed-vl pass, and a correct one needs alias analysis (@sec:decisions).],
)

#measured[
  End state: `codegen` 0.231 s — *1.59× faster than onnx-mlir `--O3`* on a
  matched back end, and within noise (1.03×) of onnx-mlir's best shipping product
  `o3emitobj`. From 1.28× behind to parity, via two committed `dr-opt` transforms
  (`powf→mul`, transcendental poly-approx) plus the `-O3` back-end flag; the GEMM
  path was already optimal.
]
