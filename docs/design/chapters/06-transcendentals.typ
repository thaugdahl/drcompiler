#import "../theme.typ": *

= Transcendentals <sec:transc>

A transformer is not only GEMMs. GELU evaluates `powf(x, 3.0)` and `tanh` per
element; softmax evaluates `exp`; LayerNorm evaluates `sqrt`. Lowered naïvely
these become *scalar libm calls* — and a `call` is a hard barrier to the loop
vectorizer. Neither `clang -O2/-O3` nor onnx-mlir `--O3` strength-reduces
`powf(x, 3.0)` to multiplies (the `math.powf` carries no fast-math flag, so the
back end leaves it a `call powf@plt`). On openai-gpt that GELU tail is ~20–45% of
runtime, paid in full by both compilers.

`dr-math-strength-reduce` removes the barrier in two steps.

#figure(
  cetz.canvas({
    import cetz.draw: *
    node((0, 0),
      align(left, text(size: 6.8pt, font: "DejaVu Sans Mono")[
        `for k {`\
        `  t = powf(x, 3.0)`  ← `call`\
        `  ...`\
        `  u = tanh(z)`        ← `call`\
        `  out[k] = ... }`
      ]),
      name: "b", kind: "loss", w: 4.3cm, h: 2.1cm)
    content((0, 1.45), text(size: 8pt, weight: 700, fill: pal.loss)[scalar libm calls — vectorizer blocked])
    node((5.5, 0.55), [① `powf(x,n)`\ → `x·x·x`\ #text(size:6.5pt)[exact, by squaring]], name: "p1", kind: "accent", w: 2.5cm, h: 1.0cm)
    node((5.5, -0.85), [② `exp`/`tanh`\ → polynomial\ #text(size:6.5pt)[poly-approx]], name: "p2", kind: "accent", w: 2.5cm, h: 1.0cm)
    arrow("b.east", "p1.west"); arrow("b.east", "p2.west")
    node((10.6, 0),
      align(left, text(size: 6.8pt, font: "DejaVu Sans Mono")[
        `for k step 8 {`\
        `  t = x·x·x`\
        `  u = poly(z)`\
        `  out = ... }`  ← vector
      ]),
      name: "a", kind: "win", w: 4.3cm, h: 2.1cm)
    arrow("p1.east", "a.west"); arrow("p2.east", "a.west")
    content((10.6, 1.45), text(size: 8pt, weight: 700, fill: pal.win)[call-free arith — auto-vectorizes])
  }),
  caption: [`dr-math-strength-reduce`. ① `powf(x, n)` with small integer `n` →
  multiplies (exponentiation by squaring, exact). ② with `poly-approx`, `exp`/
  `tanh`/… → vectorizable polynomial arith (upstream
  `populateMathPolynomialApproximationPatterns`). With the `call` barrier gone,
  the pointwise loop auto-vectorizes.],
) <fig:transc>

== Why at the MLIR level, not a backend flag

The obvious alternative is `clang -fveclib=libmvec`. It was tried (@sec:gap) and
is *insufficient*: libmvec vectorizes `exp` but has *no vector `tanh`*, and GELU
is `tanh`-dominated, so it bought almost nothing. Doing the rewrite in MLIR —
turning the transcendental into call-free arith — lets the *ordinary* loop
vectorizer (clang `-O3`, or the register-block machinery) handle it, independent
of which vector libm the back end happens to ship. This is exactly what
onnx-mlir's own `--EmitObj` back end does internally; the MLIR rewrite matches it
portably.

== Tie to the Machine Model

The vector width any of this vectorizes to is `preferredVectorElems` (@sec:rb) —
one width for GEMM and eltwise alike. And whether the eltwise tail is worth
attacking at all is a *roofline* question (@sec:roofline): the transcendental
loops are compute-heavy (a `tanh` polynomial is ~10 FMAs) and small in footprint,
so they sit on the compute ceiling — vectorizing them pays. The same reasoning
correctly *rejected* the equivalent work on gpt-neox, whose attention eltwise is
bandwidth-bound.

#measured[
  `powf → x·x·x` alone: openai-gpt codegen 278.6 → 263.7 ms (correct to
  6.8 · 10#super[−6]). `+ poly-approx`: → 239 ms (5.9 · 10#super[−6]). Together
  they move the model from 286 ms to 239 ms — and they are what the “backend gap”
  in @sec:gap actually was.
]
