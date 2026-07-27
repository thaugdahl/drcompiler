#import "../theme.typ": *

= canonicalizeAllocaGemm <sec:alloca>

The register-block vectorizer needs a *perfect* reduction band — an `i`–`j`–`k`
nest whose innermost body is only the multiply-accumulate. A tensor compiler
does not hand it one. onnx-mlir lowers every FFN and projection GEMM to a
*scalar-alloca accumulator with an inline bias epilogue*:

```mlir
for i { for j {
  %a = alloca : memref<f32>          // rank-0 scalar accumulator
  %a[] = 0
  for k { %a[] = %a[] + A[i,k]*B[k,j] }   // load/add/store %a EVERY k-step
  C[i,j] = %a[] + bias[j]            // bias epilogue, inline in the j-body
}}
```

Two things make this un-blockable. The j-body is *not* just the k-loop (there is
an init before it and a bias-add after), so the nest is *imperfect* — Stage-1b
refuses it. And the accumulator is a rank-0 `alloca` carried through memory: the
inner loop does a `load`/`add`/`store` of `%a` *every* k-step, a latency-bound
dependent chain at ~1.5 GFLOP/s. This is the form that ran openai-gpt at 20+
seconds.

== The transform

`canonicalizeAllocaGemm` (`AffineRegisterBlock.cpp`, gated on
`hasExplicitGemmModel`) rewrites this into three *perfect* nests: it promotes the
rank-0 `alloca` to the spatial output `C[i,j]` and fissions the body by segment —
`INIT` (before-k), `GEMM` (the k-loop, now a perfect `i`-`j`-`k` band), and `EPI`
(the bias-add after-k):

#figure(
  cetz.canvas({
    import cetz.draw: *
    // before
    node((0, 0),
      align(left, text(size: 6.8pt, font: "DejaVu Sans Mono")[
        `for i, j {`\
        `  alloca a; a=0`\
        `  for k { a = a + A·B }`\
        `  C = a + bias }`
      ]),
      name: "before", kind: "loss", w: 4.0cm, h: 2.2cm)
    content((0, 1.55), text(size: 8pt, weight: 700, fill: pal.loss)[imperfect · scalar `alloca` chain])
    // arrow
    node((4.9, 0), [*canonicalize-*\ *AllocaGemm*], name: "t", kind: "accent", w: 2.0cm, h: 1.0cm)
    arrow("before.east", "t.west")
    arrow("t.east", (6.4, 0))
    // after: 3 perfect nests
    let after(y, body, nm, k) = node((9.3, y), body, name: nm, kind: k, w: 4.4cm, h: 0.85cm)
    after(1.45, align(left, text(size: 6.8pt, font: "DejaVu Sans Mono")[`for i,j { C[i,j] = 0 }`]), "init", "io")
    after(0.0,  align(left, text(size: 6.8pt, font: "DejaVu Sans Mono")[`for i,j,k { C += A·B }` ← perfect]), "gemm", "win")
    after(-1.45,align(left, text(size: 6.8pt, font: "DejaVu Sans Mono")[`for i,j { C += bias[j] }`]), "epi", "io")
    content((9.3, 2.15), text(size: 8pt, weight: 700, fill: pal.win)[3 perfect nests · C in registers])
    line((6.4, 0), (7.1, 1.45), stroke: pal.line); line((6.4,0),(7.1,0), stroke: pal.line); line((6.4,0),(7.1,-1.45), stroke: pal.line)
    // downstream
    arrow("gemm.south", (9.3, -2.4))
    content((9.3, -2.65), text(size: 7pt, fill: pal.mm)[→ register-block + cache-tile crush it (@sec:rb)])
  }),
  caption: [`canonicalizeAllocaGemm`: promote the rank-0 `alloca` accumulator to
  the spatial `C[i,j]`, then fission init / k-reduction / bias into three perfect
  nests. The middle nest is the perfect `i`-`j`-`k` band the vectorizer needs;
  the bias epilogue becomes a cheap separate pass (or a fusion target).],
) <fig:alloca>

== Legality — refusing to miscompile

Fission and promotion are only sound when the output `C` is genuinely a
write-only result of this GEMM. The pass (hardened after an adversarial review)
*refuses* — leaving the scalar nest untouched and correct — unless all hold:

#callout("Guards (any failure ⇒ skip, leave scalar)", [
  - `C` is *write-only* in the j-loop — *any* load of `C` rejects (this single
    guard kills in-place `C==A`/`C==B`, beta/residual `C = acc + C`, and a
    k-loop that reads `C`).
  - exactly one ≥1-D output store, indexed by exactly the `i`,`j` IVs.
  - the stored value *derives from* the accumulator.
  - additive reductions only (`arith.add{f,i}`).
  - the pre-k segment is exactly `{alloca, invariant init-store}`.
  - the accumulator is local (used only inside the j-loop).
], fg: pal.loss, bg: pal.warnbg)

These are not hypothetical: the review built `C==A` and beta-accumulation inputs
and confirmed each is left scalar. The bench `norm-rel-err ≤ 1e-4` gate is the
backstop.

#measured[
  With `canonicalizeAllocaGemm` enabled (via `bench/zen4-gemm.json`), openai-gpt
  goes from the 20+ s scalar-alloca form to a register-blocked, vectorized GEMM.
  It is the transform that makes the deep-K transformer FFN tractable at all; the
  rest of the codegen win (@sec:gap) is the kernel it unlocks.
]
