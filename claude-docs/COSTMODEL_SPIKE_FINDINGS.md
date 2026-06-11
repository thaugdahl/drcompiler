# Cost-Model Validation Spike — Register-Tile Selection

**Date:** 2026-06-04
**Author:** Claude (Opus 4.8)
**Purpose:** Before committing to Phase 1, test the planned reframing of the contribution
("analytical register-tile selection for the affine path"). Gate: does a register-pressure
cost model analytically pick the empirically-best `(m_r, n_r)` register block, and does that
pick **shift correctly** when the register file / vector width changes?

**Verdict: NEGATIVE for the cost-model pillar.** The optimum is nearly invariant to register
count and vector width — a fixed `8×16` block is **95–100%** of the per-arch best on every
configuration tested. An analytical per-arch register-tile selector buys **≤5%**, so it is not
load-bearing. The natural architectural-register-fit model additionally **mispredicts all three**
configurations. The contribution cannot rest on the cost model; it must rest on the
register-blocking **transformation** itself (fixed/heuristic tile).

---

## Method

Exhaustive `(m_r, n_r)` sweep of the Phase-0 hand-written register-blocked affine GEMM
(`gen_regblock.py`), N=1024 (L3-resident, compute-bound), Ryzen 7950X3D **core 8**
(non-V-cache CCD, 32 MB L3), median of 3×5 reps, checksums verified, lowered through the
canonical pipeline, compiled `clang -O3 -ffast-math`.

**Controlled "arch" knob = ISA level** (changes the model's inputs on ONE silicon, no remote box):

| label | clang flag | vector width V | usable vector regs Nreg | what it isolates |
|---|---|---|---|---|
| **v3** | `-march=x86-64-v3` | 4 (YMM, verified no ZMM) | 16 | scarce registers |
| **v4** | `-march=x86-64-v4` | 4 (clang kept YMM0–31) | 32 | +register count only |
| **native** | `-march=native` | 8 (ZMM) | 32 | +vector width |

v3→v4 isolates register *count* (width fixed at 4). v4→native adds width.
**Integrity check:** disassembled binaries — v3/v4 contain only `ymm`, native uses `zmm`.
(An earlier `-mavx2 -mfma -mno-avx512f` attempt **leaked ZMM** — 372 zmm uses — and was discarded.)

## Empirical sweeps (GFLOPs, N=1024 core8)

MR-row best (best NR per MR):

| arch (V,Nreg) | MR1 | MR2 | MR4 | **MR8** | MR16 |
|---|---|---|---|---|---|
| v3  (4,16) | 17@n32 | 23@n32 | 30@n32 | **35@n32** | 28@n32 |
| v4  (4,32) | 17@n32 | 29@n32 | 34@n32 | **43@n16** | 30@n32 |
| native(8,32)| 10@n32 | 32@n32 | 50@n32 | **51@n16** | 33@n8 |

- **MR=8 is the universal optimum.** MR=16 cliffs on every arch. MR=4 is second.
- Best block per arch: v3 `8×32`=35.3, v4 `8×16`=43.3, native `8×16`=50.9.

## Model vs empirical

Architectural register-fit model: `regs(m,n) = m·⌈n/V⌉ (acc) + ⌈n/V⌉ (B) + m (A-broadcasts)`,
pick the largest `m·n` block with `regs ≤ Nreg`.

| arch | empirical best | model pick | model perf | match? | **fixed 8×16** | % of best |
|---|---|---|---|---|---|---|
| v3 | 8×32 = 35.3 | 2×16 | 22.3 | **NO** | 33.6 | **95.2%** |
| v4 | 8×16 = 43.3 | 2×32 | 29.2 | **NO** | 43.3 | **100%** |
| native | 8×16 = 50.9 | 4×32 | 50.3 | **NO** | 50.9 | **100%** |

The model mispredicts all three — it under-sizes the block, because **architectural** register
count is not the binding constraint: Zen4 has ~192 *physical* vector registers, OoO rename plus
cheap L1-resident spills absorb moderate over-subscription. Tellingly, the **scarcest-register**
arch (v3) peaks at the *largest* NR (`8×32`) — the **opposite sign** to the register-fit model's
"fewer registers → smaller block."

## What actually governs the optimum

Two-sided, neither term is register-count:
1. **Lower edge (ILP floor):** need ≈ `FMA_latency × FMA_units` independent accumulator chains in
   flight (Zen4: 4 cyc × 2 units = 8). Small blocks starve the FMA units → the MR1/MR2 rows are slow.
2. **Upper edge (broadcast/decode pressure):** MR=16 over-subscribes A-broadcasts + acc and cliffs
   on every arch.
   The knee sits at **MR=8** independent of (V, Nreg). NR≥16 to amortize.

## Verdict and consequence

- The "analytical register-tile selection" reframing **does not survive**: over the axes a cost
  model would key on locally (register count, vector width), the optimum barely moves, and a
  constant `8×16` is within 5% everywhere. The cost model is **not load-bearing**.
- This is, however, **good for the transformation pillar**: register-blocking with a *fixed* `8×16`
  reliably captures the large win (35–51 GFLOPs, 70–107× over naive affine) across register files
  and vector widths. Phase 1 can hardcode/heuristic the tile and be robust — no per-arch model needed.

## The one remaining shot for a load-bearing model (untested)

I varied only `Nreg` and `V` (ISA knob). I did **not** vary FMA latency/throughput — that needs a
physically different microarch. The MR sweet spot is set by the ILP floor (`lat × units`); on a core
with different FMA latency or a single FMA unit (e.g. Broadwell-EP, FMA lat 5; or in-order/ARM),
the MR knee *could* move off 8, and an **ILP/latency** model (not register-fit) might predict it.
That is the only experiment that could revive a selector — and it requires the Broadwell box
(`spike-H-xeon-pkg`). Prior given the local degeneracy: low. If Broadwell also peaks at MR=8, the
cost model is fully dead and the contribution is purely the transformation.

## Reproduce
`/tmp/claude/rbsweep.sh <N> <label> <clang-arch-flags>`; sweeps in
`sweep_v3.csv`, `sweep_v4.csv`, `sweep_avx512.csv` (=native). Generator `gen_regblock.py`,
harness `gemm_main.c`. The discarded contaminated run: `sweep_avx2.csv` (ZMM leak — do not use).
