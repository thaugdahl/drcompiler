# Phase 1 — Automated Register-Blocking Pass (`affine-register-block`)

**Date:** 2026-06-04
**Author:** Claude (Opus 4.8)
**Status:** GATE PASSED. The pass automatically reproduces the Phase-0 hand-built
register-blocked GEMM, including on **real cgeist-lowered** output.

---

## What was built

A new MLIR pass `affine-register-block` (`lib/Transforms/AffineRegisterBlock.cpp`,
registered in `Passes.td`, header `AffineRegisterBlock.h`). It brings DPS-style
register-blocking to the **memref/affine** path (cgeist, ONNX-MLIR Krnl→Affine,
MARCO) — the audience pinned by `RISK_PROBE_FINDINGS.md` where tensor-Linalg's
bufferization mechanism cannot reach.

Given a perfectly-nested affine band whose innermost loop carries a memory
reduction with a loop-invariant accumulator address (the `C[i,j] += A[i,k]*B[k,j]`
GEMM shape), the pass:

1. **unroll-and-jams** the two outer loops by `mr × nr` (default 8×16) using
   upstream `affine::loopUnrollJamByFactor` — on the memory form, where it is
   fully supported (no `iter_args` yet);
2. **promotes** each of the `mr*nr` accumulators from a load/store pair into an
   `affine.for` `iter_args` SSA value carried across the reduction loop, via
   `AffineForOp::replaceWithAdditionalYields`. The initial load is hoisted before
   the loop (cloning the `affine.apply` index offsets that unroll-and-jam emits
   inside the body) and the final store is sunk after the loop.

The `(mr, nr)` tile is a **fixed knob**, justified by `COSTMODEL_SPIKE_FINDINGS.md`
(8×16 within 5% of the per-arch optimum across register files + vector widths).

## Gate results (N=1024, Ryzen 7950X3D core 8, `-march=native -ffast-math`, checksums identical)

| input | GFLOPs | vs naive |
|---|---|---|
| naive affine (canonical template) | 0.48 | 1× |
| **canonical + `affine-register-block{mr=8 nr=16}`** | **51.5** | **107×** |
| Phase-0 hand-built (`gen_regblock.py`) | 50.5 | 105× |
| **real cgeist gemm** (`cgeist --raise-scf-to-affine -O2`) | 0.47 | 1× |
| **real cgeist gemm + pass** | **50.9** | **108×** |

All checksums = `1.278750e+04` (correctness verified against the naive baseline).
**The automated pass matches — slightly beats — the hand-built Phase-0 number, and
does so on cgeist's actual output, not just the synthetic template.** This is the
Phase-1 gate from `CODEGEN_CAMPAIGN.md`.

## Validation

- Lit regression test: `test/AffineRegisterBlock/gemm-register-block.mlir`
  (FileCheck on the 2×2 transform: outer loops stepped, accumulators hoisted,
  reduction loop carries `iter_args` with no in-loop store, finals sunk). PASS.
- Full suite: `ninja -C build check-drcompiler` — 182 pass, 16 unresolved
  (all pre-existing ONNX/mnist + resnet50 intermediate tests needing external
  model downloads; unrelated to this change). No regression.

## Scope / limitations (honest)

- **Detection is intentionally narrow:** a perfect 3-deep nest whose innermost
  loop has a k-invariant load/store accumulator. The clean cgeist gemm matches
  exactly. The *PolyBench* gemm under the repo's DCE-prevention hack does **not**
  (cgeist constant-folds A/B into inline arithmetic and the C address degenerates)
  — that artifact is not a representative kernel; a clean standalone gemm is.
- **Imperfect outer nests not yet handled.** A real PolyBench gemm wraps the
  matmul i-loop with a sibling `C *= beta` scaling loop (imperfect nest); the
  current detector requires `onlyChildFor`. `loopUnrollJamByFactor` itself handles
  imperfect nests, so the extension is: detect the innermost reduction loop and
  unroll-jam its enclosing loops directly rather than requiring a perfect band.
  Next step if generalizing beyond the clean-gemm gate.
- Fixed tile, single element type (f64), single reduction per nest. Multi-output
  / fused-epilogue kernels untested.

## Reproduce
- Pass: `build/tools/dr-opt/dr-opt --pass-pipeline='builtin.module(func.func(affine-register-block{mr=8 nr=16}))'`
- Gate harness: `/tmp/claude/phase1_gate.sh 1024 8 16` (naive vs pass vs hand).
- cgeist gemm: `/tmp/claude/cg_gemm.c` → cgeist (docker `drcc:x86_64`) →
  `cg_clean.mlir`; run/perf inline in session log.
