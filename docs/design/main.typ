//===-- main.typ — drcompiler design documentation ------------------------===//
// Build:  typst compile docs/design/main.typ docs/design/drcompiler-design.pdf
// (run from the repo root so the @preview/cetz package and theme.typ resolve).

#import "theme.typ": *
#show: setup

// ---------------------------------------------------------------- title page
#set page(numbering: none)
#v(3cm)
#align(center)[
  #text(size: 30pt, weight: 800, fill: pal.mm)[drcompiler]
  #v(2pt)
  #text(size: 15pt, weight: 600, fill: pal.ink)[
    A machine-model-driven codegen & recomputation compiler
  ]
  #v(4pt)
  #text(size: 11pt, fill: pal.muted)[
    Concepts · Decisions · Reasoning · How it works
  ]
  #v(1.4cm)
  #block(width: 80%, align(left,
    text(size: 10pt, fill: pal.muted)[
      Everything in this document hangs off one object — the *Machine Model*
      (`drcompiler::MachineModel`). It is the single resolved description of the
      target hardware; every cost-modelled pass asks it the same questions and
      gets answers consistent with one machine. The chapters move outward from
      that spine: the cost model and roofline it powers, the codegen transforms
      it configures, the recomputation analyses it prices, and a measured case
      study (closing the openai-gpt gap to onnx-mlir's `--O3`) that shows the
      reasoning end to end.
    ]))
]
#v(1fr)
#align(center, text(size: 9pt, fill: pal.muted)[
  Generated design reference · illustrations in CETZ · figures are normative of
  the code they describe.
])
#pagebreak()

// ---------------------------------------------------------------- front matter
#set page(numbering: "i")
#counter(page).update(1)
#outline(title: [Contents], indent: auto, depth: 2)

#set page(numbering: "1")
#counter(page).update(1)

// ---------------------------------------------------------------- chapters
#include "chapters/01-overview.typ"
#include "chapters/02-machine-model.typ"
#include "chapters/03-cost-model-roofline.typ"
#include "chapters/04-register-block.typ"
#include "chapters/05-canonicalize-alloca-gemm.typ"
#include "chapters/06-transcendentals.typ"
#include "chapters/07-recomputation-fission.typ"
#include "chapters/08-gap-analysis.typ"
#include "chapters/09-decisions.typ"
#include "chapters/A-pass-internals.typ"
