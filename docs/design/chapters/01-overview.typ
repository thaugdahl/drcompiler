#import "../theme.typ": *

= Overview

*drcompiler* is an out-of-tree MLIR compiler with two jobs that share one
foundation. The first job is *cache-aware data recomputation*: deciding, per
load, whether to re-derive a value from its producers instead of paying a
memory round-trip. The second is *machine-model-driven codegen*: turning the
naïve affine loop nests a tensor compiler emits into register-blocked,
vectorized, cache-tiled kernels. Both jobs are decisions about the same scarce
resources — cache capacity, memory bandwidth, vector registers, FMA pipes — so
both consult the same description of the machine.

That description is the *Machine Model* (`drcompiler::MachineModel`,
@sec:mm). It is the spine of this document: every later chapter is a way of
*using* it. Keep one sentence in mind throughout —

#keyidea[
  There is *one* resolved description of the target machine, and every
  cost-modelled pass asks it the same questions. A decision is only as portable
  as the model it was derived from; hard-coded knobs are the bug the Machine
  Model exists to remove.
]

== The decoupled pipeline

drcompiler is not a monolith. The front end (`cgeist`, from Polygeist, built on
*LLVM 18*) and the optimizer (`dr-opt`, built on *LLVM 22*) never share a
library — they communicate through *textual `.mlir` files*. This is a
deliberate architectural decision (@sec:decisions): it lets the optimizer track
upstream LLVM independently of the front end, at the cost of one serialization
seam that a small `sed` fixup bridges (the DLTI integer-width change between the
two LLVM versions).

#figure(
  cetz.canvas({
    import cetz.draw: *
    let y = 0
    node((0, y),   [source\ `.c`],            name: "src", kind: "io",  w: 1.7cm)
    node((2.6, y), [cgeist\ (LLVM 18)],       name: "cg",  kind: "pass", w: 2.1cm)
    node((5.2, y), [`.mlir`\ +DLTI `sed`],    name: "mlir",kind: "io",  w: 2.0cm)
    node((8.4, y), [*dr-opt* (LLVM 22)],      name: "dr",  kind: "pass", w: 2.7cm)
    node((11.6, y),[`mlir-opt`\ lower],       name: "lo",  kind: "pass", w: 1.9cm)
    node((11.6, y - 2.0), [`mlir-translate`\ → LLVM IR], name: "tr", kind: "pass", w: 2.4cm)
    node((8.4, y - 2.0),  [`clang` `-O3`\ `-march=native`], name: "cl", kind: "pass", w: 2.4cm)
    node((5.2, y - 2.0),  [object\ `.o`],     name: "obj", kind: "io",  w: 1.7cm)
    arrow("src.east", "cg.west")
    arrow("cg.east", "mlir.west")
    arrow("mlir.east", "dr.west")
    arrow("dr.east", "lo.west")
    arrow("lo.south", "tr.north")
    arrow("tr.west", "cl.east")
    arrow("cl.west", "obj.east")
    // decoupling boundary
    line((6.8, 1.1), (6.8, -3.0), stroke: (paint: pal.accent, dash: "dashed", thickness: 1pt))
    content((6.8, 1.45), text(size: 7.5pt, fill: pal.accent, weight: 700)[LLVM 18 ∥ 22 seam])
    // dr-opt internals
    content((8.4, -3.25), text(size: 7.5pt, fill: pal.pass)[
      raise-malloc-to-memref · data-recomputation · memory-fission · codegen
    ])
  }),
  caption: [The decoupled pipeline. `cgeist` (LLVM 18) and `dr-opt` (LLVM 22)
  communicate only through textual `.mlir`; the dashed line is the version seam
  the DLTI `sed` fixup bridges. All cost-modelled work lives in `dr-opt`.],
) <fig:pipeline>

== The pass families

`dr-opt` registers three custom passes plus the codegen transforms, in two
families that map to the two jobs:

#grid(columns: (1fr, 1fr), gutter: 12pt,
  callout("Recomputation family", [
    - `raise-malloc-to-memref` — undo Polygeist's malloc/GEP/LLVM-load form back
      to `memref` ops so the analyses see structured access.
    - `data-recomputation` — interprocedural load/store provenance; replace a
      *single-provenance* load with its recomputed value when the cache cost
      model says it pays (@sec:dr).
    - `memory-fission` — the inverse of loop fusion: split fused sibling loops
      and *materialize* a shared expensive subexpression to a buffer when that
      is cheaper than recomputing it N times (@sec:fission).
  ], fg: pal.io, bg: pal.iobg),
  callout("Codegen family", [
    - `dr-scalar-reduction-demote` / `-promote` — bracket the vectorizer: put
      reductions into the memref-accumulator form it matches, then lift any
      leftovers back to register `iter_args` (@sec:rb).
    - `affine-register-block` — the GEMM/conv engine: register-block + vectorize
      + cache-tile, all sized by `MachineModel::gemmBlocking()` (@sec:rb).
    - `canonicalizeAllocaGemm` — recover a perfect, blockable band from a tensor
      compiler's scalar-accumulator GEMM (@sec:alloca).
    - `dr-math-strength-reduce` — `powf(x,n)` → multiplies and (optionally)
      transcendentals → vectorizable polynomials (@sec:transc).
  ], fg: pal.pass, bg: pal.passbg),
)

Every box above that makes a *sizing* or *profitability* decision routes it
through the Machine Model. The next chapter is that model.
