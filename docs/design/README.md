# drcompiler — design documentation

A set of Typst documents (illustrated with [CETZ](https://github.com/cetz-package/cetz))
that explain drcompiler's concepts, decisions, reasoning, and "how it works",
all spined on the **Machine Model** (`drcompiler::MachineModel`).

## Build

```sh
# from the repo root (so @preview/cetz and theme.typ resolve)
typst compile docs/design/main.typ docs/design/drcompiler-design.pdf
# live preview while editing:
typst watch docs/design/main.typ docs/design/drcompiler-design.pdf
```

Requires Typst ≥ 0.14 and network access on first run (to fetch
`@preview/cetz:0.3.4` into the local package cache).

## Layout

| file | contents |
|------|----------|
| `main.typ` | title, contents, chapter includes |
| `theme.typ` | palette, page setup, callouts, and the cetz `node`/`arrow`/`trot` helpers every figure uses |
| `chapters/01-overview.typ` | what drcompiler is + the decoupled pipeline |
| `chapters/02-machine-model.typ` | **the spine** — fields, resolution contract, the hub |
| `chapters/03-cost-model-roofline.typ` | the two roofline arms + the ridge |
| `chapters/04-register-block.typ` | the GEMM kernel, micro-kernel tile, accumulator-chain ILP |
| `chapters/05-canonicalize-alloca-gemm.typ` | recovering a perfect band from a scalar-alloca GEMM |
| `chapters/06-transcendentals.typ` | `powf→mul` + poly-approx |
| `chapters/07-recomputation-fission.typ` | provenance, recompute, memory fission |
| `chapters/08-gap-analysis.typ` | case study: closing the openai-gpt gap to onnx-mlir `--O3` |
| `chapters/09-decisions.typ` | standing decisions, the spike-first method, measured no-gos |

## Convention

Figures are meant to be *normative* of the code they describe: if a figure and
the source disagree, the figure is the bug. Each chapter cross-references the
Machine Model fields/methods it relies on, so the document and
`include/drcompiler/Analysis/MachineModel.h` should be read together.
