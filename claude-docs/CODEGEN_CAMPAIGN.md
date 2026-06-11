# Codegen Campaign — Closing the MLIR Affine→LLVM Register-Blocking Gap

**Date:** 2026-06-03
**Status:** Scoping. Not started. Phase 0 is a hard go/no-go spike — do not invest beyond it without a pass.

---

## 0. Why this exists (the problem, with evidence)

The tiling/fusion investigation (`TILING_INVESTIGATION_FINDINGS.md`) ended on a wall that is *itself* the most interesting thing it found:

> The MLIR affine→LLVM lowering produces code **3.5–4.6× slower than clang‑on‑C** for GEMM, **at every tile size**, and standard MLIR passes don't fix it.

Measured root cause (real cgeist PolyBench gemm, LARGE, Ryzen 7950X3D core8, `clang -O3 -march=native -ffast-math` backend for both):

| | instructions | time | flops/instr | bottleneck |
|---|---|---|---|---|
| clang-on-C | 590 M | 0.087 s | 4.5 | register-blocked, vectorized |
| MLIR affine path | 2,745 M | 0.300 s | 0.96 | **1:1 load:FMA — no register blocking** |

The MLIR loop **is** vectorized (width-8 zmm); it just reloads `A`/`B` operands for every FMA because there is **no register tile** (no `m_r×n_r` block of `C` accumulators held in registers and reused across loads). clang/BLIS/Polly all do this via unroll-and-jam + accumulator promotion; the affine→LLVM path does not, and the off-the-shelf passes fail to add it:

- `affine-scalrep` → no-op
- `affine-super-vectorize` → emits 0 vector ops
- `affine-loop-unroll-jam` ×4/×8 → no change
- `affine-loop-unroll` ×8 → 2.7× **worse**

**This gap caps every affine-dialect compiler** (cgeist/Polygeist, general C→affine, polyhedral output) below the production compiler on reduction-heavy code — independent of any cost model.

---

## 1. Thesis and contribution

> **A cost-model-driven register-blocking codegen path for the MLIR `affine` dialect that brings reduction kernels (GEMM, SYRK, conv, contractions) to clang/BLIS-class performance without leaving affine, without autotuning, and without hand-written micro-kernels — by analytically choosing the register tile from a unified register cost model.**

What makes it defensible:
- **Real, uncontested gap.** The affine→LLVM path is non-competitive on reductions; the whole affine ecosystem inherits it.
- **It uses the project's actual asset** (the register cost model) for the thing register modeling is *good* at — sizing an `m_r×n_r` register tile to fit the vector register file — not the thing it's bad at (spill-count prediction, which the canary already flagged).
- **It does not compete with Polly head-on.** The claim is "make MLIR's *own* native lowering competitive," not "beat a polyhedral tiler." Polly/clang/BLIS become the **target/ceiling**, not the adversary.
- **Analytical, not autotuned** — the differentiator vs TVM/Ansor/autotuners.

### Honest positioning vs prior work (decides the framing)
- **BLIS / MKL** — hand-written micro-kernels. The *ceiling*; not a compiler. We aim to approach them analytically.
- **Polly** — polyhedral tiling; its register-level codegen still leans on the LLVM backend. Compare against it; expect it to be strong on GEMM.
- **Linalg + vector dialect + IREE / Triton** — **already** do register-blocked vectorized codegen — *for the structured tensor abstraction.* **This is the central reviewer question:** *"why not just use Linalg?"* The honest answer, and the scoping boundary of this whole campaign: **the contribution is for the `affine` path** — programs from C frontends (cgeist/Polygeist), polyhedral tools, and hand-written affine that are **not** in Linalg form and cannot be cheaply raised to it. If a kernel is trivially expressible in Linalg, this work has no reason to exist. **Phase 0 must therefore also justify that the affine-path audience is real and non-empty** (it is: cgeist output, ONNX-MLIR Krnl→Affine, MARCO `bmodelica`→affine — all affine, not Linalg).

---

## 2. The technical crux

A register-blocked micro-kernel for a reduction (e.g. GEMM inner) requires, at the affine level, the combination that no single existing pass produces:

1. **Unroll-and-jam** the parallel band dims by `(m_r, n_r)` → `m_r·n_r` independent accumulator instances.
2. **Promote the accumulators to SSA values** carried as `iter_args` across the reduction loop (so they live in registers, not memory). *This is the step `affine-loop-unroll-jam` omits and why it didn't help — it unrolls but leaves `C[i][j]` in memory.*
3. **Leave the inner reduction in a form LLVM vectorizes** (or emit `vector` dialect explicitly). From the data, the LLVM backend already vectorizes affine-lowered loops width-8 — so if (1)+(2) land, vectorization may come for free; verify, don't assume.

The pass sequence is roughly: `tile (cache) → unroll-jam (register, m_r×n_r) → accumulator scalar-replacement/promotion → (optional explicit vectorize) → lower`. The novel pass is the **register-block + accumulator-promotion** step; the cost model chooses `m_r, n_r`.

---

## 3. Phases (with hard go/no-go gates)

### Phase 0 — De-risking spike (DAYS, do this first, GATE)
**Question: can a register-blocked affine GEMM reach clang/BLIS through the existing affine→LLVM lowering at all?**
- By hand (or with a throwaay transform), construct a GEMM with an explicit `m_r×n_r` register tile and `iter_args` accumulators in affine, lower it through the *current* pipeline, and measure.
- **GATE:** must reach **≥70% of clang -O3** (and credibly trend toward BLIS) on gemm-LARGE. 
  - If yes → the lowering *can* express competitive code; the campaign is about *automating* it. Proceed.
  - If no (the lowering itself mangles register-blocked code, or LLVM won't vectorize it) → **stop.** The problem is deeper than codegen-pass-level and the thesis is not viable as framed.
- Also confirm a non-empty affine-path audience (cgeist/ONNX-MLIR/MARCO emit affine, not Linalg) and that those kernels are *not* trivially Linalg-raisable.

### Phase 1 — Register-block + accumulator-promotion pass (WEEKS)
- Implement the pass: given a tiled reduction band, unroll-and-jam parallel dims by `(m_r,n_r)` and promote the reduced accumulators to `iter_args`/SSA across the reduction loop.
- Start with perfectly-nested GEMM; get it automated to the Phase-0 hand result.
- **Gate:** automated pass matches the Phase-0 hand-built number on gemm.

### Phase 2 — Cost-model-driven register-tile selection (WEEKS)
- Choose `(m_r, n_r)` from the unified register model: fit `m_r·n_r` accumulators + operand vectors in the architectural vector register file (per-arch: AVX2/AVX-512/NEON via the existing `ArchHandler`).
- **This is where the project's register cost model becomes load-bearing and *correctly scoped*** (register-tile fit, not spill prediction).
- **Gate:** the analytically chosen `(m_r,n_r)` is within a few % of the best `(m_r,n_r)` from an exhaustive sweep, across ≥3 kernels and ≥2 arches. (Validate the model the way the tiling work failed to.)

### Phase 3 — Vectorization quality (WEEKS)
- Confirm/ensure the register-blocked inner product vectorizes well (rely on LLVM backend if sufficient; otherwise emit `vector` dialect FMAs explicitly). Close any remaining instruction-count gap to clang.
- **Gate:** instruction count within ~1.5× of clang on gemm; flops/instr ≥ ~3.

### Phase 4 — Generalization (WEEKS)
- Extend to syrk, syr2k, 2mm/3mm (chained), doitgen (contraction), and conv (the ONNX-relevant one). Add operand **packing** if the strided access (e.g. gemm's column `B`) needs it for the register kernel to hit bandwidth.
- Stencils need **no** register blocking (no reduction) — they keep the existing cache-tiling path. Keep the two paths separate (reduction → register-block; stencil → cache-tile).
- **Gate:** ≥4 reduction kernels within a target fraction of clang/Polly; no regressions on the stencil path.

### Phase 5 — Evaluation & write-up (WEEKS)
- Baselines: **current MLIR affine lowering** (the gap we close), **clang -O3** (the target), **Polly** (polyhedral peer), **BLIS/MKL** (hand-tuned ceiling).
- Workloads: PolyBench reductions + at least one *affine-path-native* consumer (ONNX-MLIR Krnl→Affine conv/matmul epilogues, and/or MARCO residual/Jacobian loops) to make the "affine path, not Linalg" audience concrete.
- Claim shape: *"the MLIR affine path reaches X% of clang/BLIS on reductions, up from ~25% today, with analytically-chosen register tiles — portable, no autotuning."*

---

## 4. Risks (honest) and mitigations

| Risk | Severity | Mitigation / gate |
|---|---|---|
| **Phase-0 fails** — lowering can't express competitive register-blocked code | **Fatal** | That's exactly what Phase 0 is for. Kill early. |
| **"Why not Linalg?"** — reviewers see it as reinventing IREE's codegen | High | Scope to the affine-path audience (cgeist/ONNX/MARCO); show those kernels aren't cheap Linalg. State it up front. |
| **Match, don't beat** — best case is parity with clang/BLIS | Medium | Frame as "make the affine path competitive + analytical (no autotuning)", a systems/portability contribution, not a speed record. |
| **Register model still inaccurate** — picks wrong `(m_r,n_r)` | Medium | Phase-2 gate validates against an exhaustive sweep; register-*fit* is a far easier, more reliable use of the model than spill prediction. |
| **Packing/layout needed** — strided operands (gemm B) bottleneck the kernel | Medium | Phase 4 adds packing; it's well-understood (BLIS). |
| **Generalization stalls** — works for GEMM, not conv/contractions | Medium | Phase-4 gate; if only GEMM-class works, narrow the claim honestly. |
| **Scope = a PhD chapter, not a sprint** | — | The phased gates let you stop at any failed gate with a publishable negative or a narrowed positive. |

---

## 5. Success criteria (what "done" means)
- **Minimum publishable:** the affine path reaches **≥80% of clang -O3** on GEMM-class reductions with an analytically-chosen register tile, validated predicted-vs-best across ≥2 arches, on real frontend-lowered (cgeist) code.
- **Strong:** ≥4 reduction kernels at ≥80% of clang, approaching BLIS on GEMM, demonstrated on an affine-path-native workload (ONNX/MARCO), beating Polly where Polly's heuristics misfire.
- **Negative-but-useful:** Phase 0 or 2 fails → a rigorous "why MLIR's affine path can't be made register-competitive without leaving the dialect" result.

---

## 6. Immediate next action
Run **Phase 0** as a bounded spike (days): hand-construct a register-blocked affine GEMM, push it through the current lowering, measure vs clang/BLIS. It is the single gate that determines whether any of the rest is worth starting. I can do this next.
