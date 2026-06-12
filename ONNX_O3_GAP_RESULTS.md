# ONNX_O3_GAP_RESULTS — outcome of the WP-G1..G4 campaign

Companion to `ONNX_O3_GAP_STEPS.md` (the spec).  All numbers: resnet50-v2-7,
batch 1, AMD Ryzen 9 7950X3D (Zen4), single-thread, `scripts/onnx-codegen-bench.sh`
median, back-to-back (thermal fairness), norm-rel-err ≤ 1e-4, top-1 = 858.

## Scoreboard (median-of-11, one session)

| config | median (s) | vs none | vs o3 |
|---|---|---|---|
| none (no dr-opt) | 2.086 | 1.00x | 0.62x |
| codegen `--vl 16` (pre-campaign shipped default) | 1.500 | 1.39x | 0.86x (loses) |
| **codegen — G1+G2+G3 (shipped now)** | **0.929** | **2.25x** | **1.39x (wins)** |
| onnx-mlir --O3 --EmitObj | 1.291 | 1.62x | 1.00x |

The pre-campaign default (`--vl 16`) **lost** to o3 (0.86x).  The campaign ends at
**1.39x faster than o3** — a 1.61x swing, all correctness-preserving (top1=858,
norm-rel-err 1.3e-6 throughout).

Per-step (each measured back-to-back the same session it landed):

| step | commit | what | vs o3 |
|---|---|---|---|
| G1 | `onnx_codegen_g1a` | portable cost-model VL (ship vl=8) | 1.07x |
| G2a | `onnx_codegen_g2a` | re-find Stage 3 reductions by acc memref (16 GEMMs) | 1.26x |
| G2b | `onnx_codegen_g2b` | multi-accumulator promote (safety net) | 1.26x (inert) |
| G3 | `onnx_codegen_g3a` | per-band conv VL (7x7 interiors at VL=4) | 1.39x |
| G4 | `onnx_codegen_g4a` | eltwise-fusion **measure-first → NO-GO** | 1.39x |

## WP-G1 — portable cost-model VL (commit g1a)

The `--vl 8` win over `--vl 16` is **not** the GEMMs.  Decoupling conv-vl from
the GEMM vl in a cross-build: GEMM zmm(16)->ymm(8) is within noise (-0.008 s);
the whole +0.31 s is the 5x 14x14 conv interiors clearing `interiorWidth >= VL`
at VL=8 — a coverage effect.  GEMMs are VL-indifferent on Zen4 (AVX-512 is
double-pumped over 2x256 FP pipes, so ymm and zmm have equal peak FLOPs).

Shipped a portable vector-execution model in `MachineModel`
(`vectorBitsNative` / `vectorBitsArch` / `vecRegBudget` / `avx512FreqThrottle` +
`preferredVectorElems`): vl = native datapath width raised to a register-pressure
floor.  Reproduces every measured point (Zen4 f32->8 f64->8 == old default;
Xeon-Gold f32->16 f64->8), JSON-gated so the default machine is byte-identical.
See `claude-docs/COSTMODEL_PORTABILITY_FINDINGS.md` (resolves gap #1, + the
**Idun (56-core Xeon Gold) spike plan** to validate the native-512 VL choice and
the AVX-512 frequency-licensing field).

## WP-G2 — jammed-scalar GEMM recovery (commits g2a, g2b)

Root cause (extract failing N=49 band + DRG2-instrument the Stage 3 loop on the
demote output): after mr-jamming a band's sOut, Stage 3 re-found the reduction to
vectorize with the **global** `findReductionLoopUnder(func)`, which mispairs
across resnet50's repeated same-shape residual-block GEMMs — once a band is
stranded it shadows the global walk and every following sOut targets it, a
cascade that left **16 of 33** 1x1-conv GEMMs jammed-scalar accumulating in DRAM
(8 load/store pairs per k-iteration).

g2a fix: re-find by the band's **accumulator memref** (unique per band, stable
across the jam) — `findReductionLoopUnder(func, accFilter)`.  17 -> 33 GEMMs
vectorized; 1.207 -> 1.025 s.  g2b: generalized the promote pass from 1 to N
accumulators so any *future* band that slips vectorization degrades to registers,
not DRAM (inert on resnet50 today — byte-identical — but defense-in-depth).

## WP-G3 — per-band conv VL (commit g3a)

`pickConvVL(extent, maxVL)`: the largest power-of-two VL in {4,8,..,maxVL} fitting
the conv interior, instead of bailing when the interior is narrower than the
machine VL.  resnet50's two deepest 3x3 convs (7x7 spatial, interior 4) vectorize
at VL=4 (24 vector<4xf32>, 2 step-4 bands); 56/28 unchanged.  `maxVL` is the
WP-G1 machine ceiling, so it composes with the portable model.  Scoped to the
conv stage (the GEMM path is byte-identical, so PolyBench is unaffected).
1.025 -> 0.929 s (+0.09 s, more than the spec's ~0.05 s estimate).

## WP-G4 — eltwise/BN fusion: measured NO-GO (commit g4a)

Spec gate: measure the eltwise-nest cost first; if < 0.1 s, STOP.

Measurement (`scripts/onnx-strip-eltwise-probe.py` removes the 80 elementwise
nests — 51 ReLU + ~29 BN/residual-add — from the codegen `.dr.mlir`, rebuilt and
timed; correctness ignored for the probe):

| config | median (s) |
|---|---|
| g3 (final) | 0.923–0.931 |
| g3, eltwise nests stubbed | 0.911–0.913 |
| **eltwise bound** | **~0.012 s (3 rounds: 0.0105 / 0.0114 / 0.0200)** |

The total elementwise cost is **~1.3% of runtime — far below the 0.1 s
threshold.  STOP: do not build a fusion pass.**  Why so cheap: BatchNorm is
already folded into the conv bias (WP-O2 Case-B), so the only *separate*
elementwise work is ReLU + residual adds, which are memory-bound but modest at
batch 1 and auto-vectorized by `clang -O2 -march=native`.  Epilogue-fusion into
the conv/GEMM kernels (the v2) is also bounded by this 0.012 s, so it is not
worth the invasiveness either.

## What remains (explicitly out of this campaign)

- **Idun / Xeon-Gold spike** — validate the portable VL model on a native-512
  part (the one open cross-machine check; see COSTMODEL_PORTABILITY_FINDINGS).
- Stride-2 convs + 7x7 stem vectorization (scalar by design; needs a gather /
  stride-2 vector kernel — stake ~0.1 s, revisit only if it ever matters).
- Batch > 1, other models, im2col/Winograd/packed-GEMM, threading (o3 here is
  also single-thread).
