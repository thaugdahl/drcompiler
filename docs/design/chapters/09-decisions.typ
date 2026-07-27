#import "../theme.typ": *

= Decisions & Reasoning <sec:decisions>

The architecture is the sum of a few decisions made deliberately, and a longer
list of things that were *tried and measured to not work*. Both are documented
here, because the discipline that produced the negatives is what makes the
positives trustworthy.

== Standing decisions

/ One machine, one description: All cache sizes, latencies, vector and compute
  parameters live in `MachineModel`; passes never carry private copies. This
  closed a real drift bug (a stale L2 default 4× too small on one path) and is
  the reason a decision derived on one machine can be re-derived on another by
  swapping a JSON.

/ Decoupled front end and optimizer: `cgeist` (LLVM 18) and `dr-opt` (LLVM 22)
  share no library — only textual `.mlir`. The optimizer tracks upstream LLVM
  independently; the cost is one serialization seam (the DLTI `sed` fixup,
  @fig:pipeline).

/ Codegen at the affine level: the GEMM/conv transforms extend
  `affine-register-block` rather than hooking a tensor compiler's own dialect.
  This keeps the textual-MLIR decoupling intact — drcompiler consumes whatever
  affine IR a front end emits, onnx-mlir included.

/ Byte-identical by default: every capability added after the original model —
  the compute and bandwidth arms, the thread model, `gemmBlocking`,
  `poly-approx`, `canonicalizeAllocaGemm` — is latched behind a `hasExplicit*`
  flag or a pass option and is *inert* until opted in. Adding power never
  perturbs an existing run. The lit suite proves it (byte-identical golden IR).

== The method: spike first, measure, gate

No transform is trusted until it moves the benchmark *and* passes the correctness
gate. The loop is deliberately cheap-experiment-first:

#figure(
  cetz.canvas({
    import cetz.draw: *
    node((0, 0),   [hypothesis\ #text(size:6.5pt)[from a model reading]], name: "h", kind: "mm", w: 2.4cm, h: 1.0cm)
    node((3.6, 0), [cheapest spike\ #text(size:6.5pt)[probe, not build]], name: "s", kind: "pass", w: 2.4cm, h: 1.0cm)
    node((7.4, 0), [measure\ #text(size:6.5pt)[median + norm-rel-err ≤ 1e-4]], name: "m", kind: "accent", w: 2.7cm, h: 1.0cm)
    node((11.0, 0.9),  [commit (local)\ #text(size:6.5pt)[never push]], name: "ok", kind: "win", w: 2.3cm, h: 0.9cm)
    node((11.0, -0.9), [revert + document\ #text(size:6.5pt)[the no-go is data]], name: "no", kind: "loss", w: 2.3cm, h: 0.9cm)
    arrow("h.east", "s.west"); arrow("s.east", "m.west")
    arrow("m.east", "ok.west"); arrow("m.east", "no.west")
    content((9.7, 0.95), text(size:7pt, fill: pal.win)[win])
    content((9.7, -0.5), text(size:7pt, fill: pal.loss)[no-go])
    arrow("no.south", (11.0, -1.7)); line((11.0,-1.7),(0,-1.7), stroke: (paint: pal.line, dash: "dashed")); arrow((0,-1.7),"h.south")
  }),
  caption: [The spike-first loop. A hypothesis comes from a Machine-Model reading;
  the cheapest experiment that can falsify it runs first; only a measured win
  that holds the correctness gate is committed (locally — never pushed). A no-go
  is reverted and *documented* — it is as valuable as a win.],
) <fig:spike>

== Measured no-gos (the discipline at work)

#table(columns: (auto, 1fr), inset: 6pt, stroke: (x,y) => (bottom: 0.3pt + pal.gridln),
  align: (left, left),
  [*tried*], [*measured outcome & reason*],
  [GEMM vl=16 / `zmm`], [*Slower.* Zen4 double-pumps 512-bit FMA over two 256-bit pipes — same FLOP rate, half the accumulators. `preferredVectorElems` was right to pick native vl=8.],
  [Single large `Kc` (no K-tiling)], [*Slower.* The ~9 MiB FFN B-panel streams from DRAM without tiling. `macroTile`'s L2-budgeted tile was right.],
  [`clang -fveclib=libmvec`], [*Insufficient.* Vectorizes `exp` but libmvec has no vector `tanh`; GELU is `tanh`-dominated. Fixed in MLIR instead (@sec:transc).],
  [Dedicated eltwise vectorizer], [*Slower + unsafe.* clang `-O3` already auto-vectorizes the call-free poly loops better than a fixed-vl pass; a 12-agent adversarial review found critical miscompiles (multi-dim stride, memref aliasing) needing real alias analysis. Built, measured, *reverted*.],
)

#nogo[
  A negative is not a failure — it is a closed door, recorded so it is not
  re-opened. The eltwise vectorizer was built correctly, lit-tested, *and* shown
  to be both slower and unsafe; reverting it (keeping only the two real wins) is
  the right outcome. The earlier "1.6× beat `--O3`" headline was likewise
  retracted when a fairness audit found it was an SSE2-baseline artifact — the
  corrected, backend-matched number (1.59×) is the one that survives.
]

== The through-line

Every chapter resolved to the same object. The roofline that chose *which* arm
to optimize, the `gemmBlocking` that sized the kernel, the cache pricing that
gated recomputation and fission, the vl that every vectorizer used, and the
case study that tied them together — all are queries against one
`MachineModel`. That is the design: *make the machine explicit, ask it one
consistent set of questions, and let measurement arbitrate.*
