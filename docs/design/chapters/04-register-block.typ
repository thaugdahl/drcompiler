#import "../theme.typ": *

= Register Blocking & the GEMM Kernel <sec:rb>

`affine-register-block` turns a naïve reduction nest into a register-blocked,
vectorized, optionally cache-tiled kernel. Every size it picks comes from
`MachineModel::gemmBlocking(M,N,K)` — the pass itself hard-codes nothing once a
GEMM model is present.

== The pass, end to end

The driver (`AffineRegisterBlock.cpp`) runs a fixed sequence, bracketed by the
two scalar-reduction passes:

#figure(
  cetz.canvas({
    import cetz.draw: *
    let s(x, body, nm, k: "pass") = node((x, 0), body, name: nm, kind: k, w: 2.05cm, h: 1.15cm)
    s(0,    [`demote`\ #text(size:6.5pt)[iter_args →\ memref acc]], "d", k: "io")
    s(2.5,  [canonicalize\ + interchange], "c")
    s(5.0,  [*canonicalizeAllocaGemm*\ #text(size:6.5pt)[if gemm model]], "a", k: "accent")
    s(7.5,  [family +\ cache-tile\ #text(size:6.5pt)[gemmBlocking]], "f")
    s(10.0, [vectorize\ #text(size:6.5pt)[µ-kernel]], "v")
    s(12.5, [`promote`\ #text(size:6.5pt)[leftovers →\ iter_args]], "p", k: "io")
    arrow("d.east","c.west"); arrow("c.east","a.west"); arrow("a.east","f.west")
    arrow("f.east","v.west"); arrow("v.east","p.west")
    content((6.25, -1.15), text(size: 7pt, fill: pal.mm)[
      `gemmBlocking()` supplies mr, nr, vl, and (mc,nc,kc) to the shaded stages
    ])
    darrow((6.25, -0.95), (7.5, -0.62))
    darrow((6.25, -0.95), (10.0, -0.62))
  }),
  caption: [The register-block sequence. `demote`/`promote` put reductions into,
  and recover them from, the memref-accumulator form the matcher expects.
  `canonicalizeAllocaGemm` (@sec:alloca) and the cache-tile/vectorize stages are
  driven by `gemmBlocking`; they fire as configured only when
  `hasExplicitGemmModel`, else the pass keeps its static defaults (byte-identical).],
) <fig:rbflow>

*Why the brackets?* The vectorizer matches a reduction written as a same-address
`load`/`store` pair (a memref accumulator). A tensor compiler emits some
reductions as register `iter_args` instead. `dr-scalar-reduction-demote` rewrites
those into the memref form; after vectorization, `dr-scalar-reduction-promote`
lifts any band the vectorizer *didn't* take back to `iter_args`, so a missed
band degrades to a register accumulator, never a DRAM round-trip.

== The broadcast micro-kernel

For the common case (C = A·B, a multiplicand stride-1 in the spatial dim — the
`Broadcast` family) the kernel holds an `mr × ⌈nr/vl⌉` grid of `vector<vl>`
accumulators in registers across the whole K reduction. Each k-step broadcasts
one scalar of A across a lane vector and fuses it with a contiguous
`affine.vector_load` of B:

#figure(
  cetz.canvas({
    import cetz.draw: *
    // A column (mr broadcasts)
    content((-0.2, 3.2), text(size: 8pt, weight: 700)[A column])
    for i in range(4) {
      node((-0.2, 2.4 - i*0.55), text(size:7pt)[`A[i+#i, k]`], name: "a" + str(i), w: 1.5cm, h: 0.42cm, kind: "io")
    }
    content((-0.2, 0.0), text(size: 7pt, fill: pal.muted)[… mr = 8 rows])
    // B row (nr = 2 vectors of vl=8)
    content((4.6, 3.2), text(size: 8pt, weight: 700)[B panel  (contiguous)])
    node((3.7, 2.6), text(size:7pt)[`vector<8>`], name: "b0", w: 1.7cm, h: 0.5cm, kind: "io")
    node((5.6, 2.6), text(size:7pt)[`vector<8>`], name: "b1", w: 1.7cm, h: 0.5cm, kind: "io")
    content((4.65, 3.0), text(size: 6.5pt, fill: pal.muted)[nr = 16 = 2 × vl])
    // C tile (mr x 2) accumulators
    content((4.6, 1.7), text(size: 8pt, weight: 700)[C accumulator tile  (in registers)])
    for i in range(4) {
      for j in range(2) {
        node((3.7 + j*1.9, 1.1 - i*0.52), text(size:6.5pt)[`acc`], w: 1.7cm, h: 0.4cm, kind: "accent")
      }
    }
    content((4.65, -1.25), text(size: 7pt, fill: pal.muted)[mr × ⌈nr/vl⌉ = 8 × 2 = 16 vector accumulators])
    // the FMA
    node((9.4, 1.0), [for k:\ `acc += `\ `bcast(A)·B`\ #text(size:6.5pt)[`fastmath` → FMA]],
      name: "fma", kind: "pass", w: 2.6cm, h: 1.5cm)
    arrow("a0.east", "fma.west"); arrow("b1.east", "fma.west")
    line("fma.west", (6.9, 0.6), stroke: (paint: pal.line, dash: "dashed"))
  }),
  caption: [The `Broadcast` micro-kernel (`RegisterBlock/Vectorize.cpp`).
  `vl = preferredVectorElems(elemBytes, mr, nr)`; the `mr × ⌈nr/vl⌉` accumulator
  grid must fit `vecRegBudget` (`canFitAccumulators`). A is broadcast, B is a
  stride-1 `affine.vector_load`, each k-step a vector FMA.],
) <fig:ukernel>

== Why 16 accumulators, not 4 — the ILP that won the gap

The accumulator count is not a tuning whim: it is *the* number of independent
FMA dependency chains, and it must cover the FMA pipeline. On Zen4 there are two
FMA issue pipes, each with ~4-cycle latency, so ~8 independent chains are needed
to keep both pipes full. `gemmBlocking` picks mr=8, nr=16, vl=8 → *16* chains;
the pipes are saturated. The same GEMM through onnx-mlir's `--O3` kernel carries
only *4* accumulators and runs latency-bound at ~half peak.

#figure(
  cetz.canvas({
    import cetz.draw: *
    // two FMA pipes
    node((6.5, 2.2), [FMA pipe 0\ #text(size:6.5pt)[~4-cyc latency]], name: "f0", kind: "pass", w: 2.4cm, h: 0.85cm)
    node((6.5, 1.0), [FMA pipe 1\ #text(size:6.5pt)[~4-cyc latency]], name: "f1", kind: "pass", w: 2.4cm, h: 0.85cm)
    // left: codegen 16 chains
    content((1.6, 3.1), text(size: 8.5pt, weight: 700, fill: pal.win)[codegen: 16 chains])
    for i in range(8) {
      circle((0.5 + calc.rem(i,4)*0.5, 2.4 - calc.quo(i,4)*0.45), radius: 0.12, fill: pal.win, stroke: none)
    }
    content((1.6, 0.95), text(size: 7.5pt, fill: pal.win)[≥ 8 ⇒ both pipes full\ ≈ 100% of peak])
    arrow((2.5, 2.1), "f0.west")
    arrow((2.5, 1.6), "f1.west")
    // right: o3 4 chains
    content((11.3, 3.1), text(size: 8.5pt, weight: 700, fill: pal.loss)[onnx-mlir --O3: 4 chains])
    for i in range(4) {
      circle((10.8 + calc.rem(i,2)*0.5, 2.4 - calc.quo(i,2)*0.45), radius: 0.12, fill: pal.loss, stroke: none)
    }
    content((11.3, 1.2), text(size: 7.5pt, fill: pal.loss)[< 8 ⇒ pipes stall\ ≈ 50% of peak])
    arrow((10.5, 2.0), "f0.east")
    arrow((10.5, 1.4), "f1.east")
  }),
  caption: [Accumulator-chain ILP. The two FMA pipes need ≥8 independent chains
  to hide their latency. The deep-K FFN GEMMs are 64% of the model's matmul
  FLOPs; this single difference (16 chains vs 4) is the dominant driver of the
  measured 1.30× same-backend win over onnx-mlir `--O3` on openai-gpt (@sec:gap).],
) <fig:ilp>

== Picking the vector width

`preferredVectorElems` starts at the native datapath width in elements
(`vectorBitsNative / bits`; on Zen4 f32 → 8) and *raises* vl only if
`mr × ⌈nr/vl⌉` would overflow `vecRegBudget`. It never exceeds the encodable
arch width. The counter-intuitive part — proven by measurement — is that a
*wider* vl does not help: at vl=16 (`zmm`) the GEMM is within noise of vl=8 and
slightly slower end-to-end, because Zen4 double-pumps the 512-bit FMA over the
same two 256-bit pipes (same FLOP rate) while *halving* the accumulator count
(@sec:decisions). The native width is the throughput-optimal one; the model
encodes exactly that.

== Cache tiling

When the working set `(M·K + K·N + M·N)·elemBytes` exceeds the budget,
`gemmBlocking` calls `macroTile()` to shrink an `(mc,nc,kc)` tile until the
per-tile set fits — and the budget is the *effective L2*, not the LLC. On a
big-LLC host the FFN B-panel already fits L3, so an LLC-budgeted tile never
fires, yet the kernel still streams ~9 MiB of B from L3 every i-pass; tiling to
L2 (the classic BLIS level) brings it resident. (Measured: forcing a single
large `kc` — no K-tiling — is *slower*, because B then streams from DRAM; the
L2-budgeted tile is correct, @sec:decisions.)
