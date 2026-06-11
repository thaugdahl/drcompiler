# Phase 0 — De-risking Spike: Register-Blocked Affine GEMM

**Date:** 2026-06-03
**Author:** Claude (Opus 4.8)
**Gate (from `CODEGEN_CAMPAIGN.md`):** a register-blocked affine GEMM must reach **≥70% of clang −O3** through the *current* affine→LLVM lowering, or the whole codegen campaign is dead.

**Verdict: GATE PASSED, decisively (245% of clang −O3). Campaign viable. Proceed to Phase 1 (automate the pass).**

---

## What was built

A throwaway generator (`/tmp/claude/gen_regblock.py`) emits a register-blocked affine GEMM directly: an `m_r×n_r` block of `C` accumulators carried as `affine.for ... iter_args(...)` SSA values across the reduction (`k`) loop, reusing `m_r` `A` loads and `n_r` `B` loads per k-step. Lowered through the **existing** pipeline (`--lower-affine → scf→cf→llvm → finalize-memref-to-llvm`, bare-ptr memref conv) → `mlir-translate` → `clang -O3 -march=native -ffast-math` backend. No new pass, no intrinsics — the question is only whether the lowering *can* express competitive register-blocked code.

## Results (AMD Ryzen 9 7950X3D, `taskset -c 8` = 32 MB-L3 CCD, median of 5)

### N=1024 (25 MB working set, L3-resident — isolates the *compute* / register-blocking axis)

| variant | GFLOPs | vs naive-affine | vs clang-ikj | checksum |
|---|---|---|---|---|
| naive affine (what cgeist emits) | 0.48 | 1× | 0.02× | ✓ |
| naive ijk clang (cache-broken) | 0.59 | — | 0.03× | ✓ |
| polly ikj (default / stripmine) | 5.76 | 12× | 0.27× | ✓ |
| polly ijk | 4.25 | 9× | 0.20× | ✓ |
| clang named-scalar 8×16 (hand-blocked) | 15.6 | 33× | 0.74× | ✓ |
| **clang ikj (natural −O3) — the honest ceiling** | **21.2** | 44× | 1.0× | ✓ |
| **MLIR affine register-block 8×16** | **51.3** | **107×** | **2.45×** | ✓ |

All checksums identical (`1.278750e+04`) — every variant computes the same C=A·B; the speedups are real, not elision.

### N=2048 (100 MB working set, DRAM-bound — shows the *memory* axis)

| variant | GFLOPs | vs clang-ikj |
|---|---|---|
| polly-ikj | 5.42 | 0.56× |
| clang-ikj | 9.69 | 1.0× |
| **MLIR register-block 8×16** | **13.62** | **1.41×** |

Register-blocking still wins DRAM-bound, but the margin shrinks 2.45×→1.41×: register blocking attacks the compute wall; at DRAM sizes the memory wall dominates and **Phase-4 cache tiling** becomes necessary to hold the gap.

### Block-size sweep (N=1024)
rb-4×8 17.7 · rb-8×8 47.0 · rb-4×16 39.2 · **rb-8×16 50.5** — 8×16 best (128 f64 accumulators → 16 zmm for C + operands, fits the 32-register AVX-512 file).

---

## The mechanism (why it works, verified)

The win is **SSA accumulator promotion**. The `m_r·n_r` block of `C` lives as `iter_args` (loop-carried SSA values), so LLVM register-allocates it and SLP-vectorizes the independent FMAs. This is exactly the step `affine-loop-unroll-jam` omits (it unrolls but leaves `C[i][j]` in memory) — see `TILING_INVESTIGATION_FINDINGS.md §3`.

**Surprising corollary (strengthens the contribution):** clang on the *optimal hand-written* named-scalar 8×16 micro-kernel reaches only **15.6 GFLOPs** — *below* even natural ikj — because clang's SLP vectorizer spills the 128-accumulator block (3,232 M instructions vs the affine path's 2,196 M for identical work). clang on the idiomatic `double acc[8][16]` array form is worse still (3.7 GFLOPs — no register promotion at all). **The affine `iter_args` form lowers to better-vectorized, lower-spill code than clang produces from the equivalent C.** The affine path is not merely "automating hand-tuning" — on this kernel it out-codegens clang's own auto-vectorizer.

---

## Honest caveats (what this does *not* show)

1. **"Beats clang" = beats clang's *auto-vectorizer*.** A hand-written AVX-512 *intrinsics* BLIS micro-kernel is the true ceiling and would match or exceed 51 GFLOPs. No optimized BLAS was available to measure it (system numpy links **reference netlib `cblas`**, ~8 GFLOPs — not a real ceiling; ignore it). 51 GFLOPs is near the single-core **L3-bandwidth roofline** for the 8×16 arithmetic intensity (256 flops / 192 B = 1.33 flop/B ⇒ ~39 GB/s effective). Going higher needs cache tiling to raise AI — the Phase-4 axis, not register blocking.
2. **L3-resident measurement.** N=1024 deliberately isolates register blocking from cache effects. The DRAM-size point (N=2048) shows the win narrowing; the full claim needs Phase-4 cache tiling for large N.
3. **Polly underperforms here** (5.4–5.8 GFLOPs, *worse* than its own clang-ikj baseline) — a known Polly fragility on simple already-vectorizable ikj. Polly's PolyBench-gemm-LARGE strength (~40 GFLOPs in `TILING_INVESTIGATION §2`) comes from cache tiling that this register-block-only kernel lacks at LARGE. The fair "why not Polly" answer is **size-dependent** and must be stated as such; do not claim a blanket Polly win.
4. **Hand-constructed affine.** This proves the lowering *can* express it. Phase 1 must *automate* the unroll-and-jam + accumulator-promotion pass and reproduce this number.

---

## Decision

- **Gate (≥70% of clang −O3): PASSED at 245%.** The affine→LLVM lowering expresses competitive register-blocked code; the campaign's premise holds.
- **Affine-path audience is real and non-empty** (the "why not Linalg" boundary): cgeist/Polygeist, ONNX-MLIR Krnl→Affine, MARCO `bmodelica`→affine all emit `affine`, not `linalg`, and these kernels are not cheap-Linalg-raisable. The 0.48-GFLOPs naive-affine number *is* what cgeist emits today — the gap is concrete.
- **Next: Phase 1** — implement the register-block + accumulator-promotion affine pass; gate is matching this hand-built 51 GFLOPs automatically on cgeist-lowered gemm.

### Reproduce
`/tmp/claude/gen_regblock.py <N> <MR> <NR>` → affine GEMM; lower with the bare-ptr pipeline in `/tmp/claude/phase0.sh`; harness `/tmp/claude/gemm_main.c` (extern `gemm`, zeroes C per rep, prints GFLOPs + checksum). Baselines: `gemm_ikj.c` (clang ceiling), `gemm_named.c` (hand-blocked), `gemm_bench.c` (naive ijk), `-mllvm -polly` (Polly).
