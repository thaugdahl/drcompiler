# ONNX_O3_GAP_STEPS — closing (and holding) the gap to onnx-mlir --O3

Status: SPEC (2026-06-12). Follow-up to ONNX_CODEGEN_SPEC.md WP-O0 and
REGISTER_BLOCK_VECTORIZER_FIX.md (WP-O2 part 3b, landed `2e..2g`), and to the
promote pass + harness (`onnx_codegen_3`, `onnx_codegen_o0`).
Methodology unchanged: spike-first, honest verdicts, one local commit per
step, NEVER push. The single gate for everything here is
`scripts/onnx-codegen-bench.sh` (median-of-5, norm-rel-err ≤ 1e-4, top-1
agreement) — kernel-level wins do not count until they move that table.

---

## 0. Where we are (all measured 2026-06-12, back-to-back, AVX-512 host)

resnet50-v2-7, batch 1, median of 5:

| config | median | vs none | vs o3 |
|---|---|---|---|
| none (no dr-opt) | 2.139 s | 1.00x | — |
| codegen vl=16 (shipped default) | 1.542 s | 1.39x | 0.86x (loses) |
| **codegen vl=8** | **1.229 s** | **1.74x** | **1.07x (wins)** |
| onnx-mlir --O3 --EmitObj | 1.321 s | 1.62x | 1.00x |

norm-rel-err 1.3e-6 in all configs; top-1 = 858 everywhere.

**Headline finding: `--vl 8` already beats onnx-mlir --O3 by 1.07x.** The
"gap" of the title is closed by a flag — but we do not understand *why*
vl=8 > vl=16 yet, the win is thin, and a third of the contraction FLOPs are
still scalar. This spec is (a) consolidate the vl=8 win honestly, (b) recover
the known scalar leftovers, ranked by measured cost.

### Coverage census (from `/tmp/ocb/codegen.dr.mlir`, vl=16 default config)

- **1x1 GEMMs: 17 of 33 vectorize** (8 @N=3136, 8 @N=784, 1 @N=196). The
  other **15 @N∈{196,49} are mr-jammed SCALAR with memref accumulators left
  in DRAM** — 8 acc load/store pairs *per k-iteration* in the jammed body
  (see §2, WP-G2). One N=196 band vectorizes while 15 same-shape ones don't —
  unexplained, diagnose before fixing.
- **3x3 convs: all 6 stride-1 56²/28² vectorize** (WP-O2 3b). At vl=8 the
  **5× 14² interiors also vectorize** (`1 to 9 step 8`); the 2× 7²
  (interior width 4) need VL=4. Stride-2 convs + 7x7 stem: scalar by design.
- **99 scalar f32 iter_args bands** = promote-pass output (borders/tails,
  sub-VL convs, stride-2) — these are *fine* (register accumulator), they are
  the correct scalar form.
- Eltwise/BN/ReLU: untouched, ~51 separate nests, each a >L2 tensor
  round-trip (ONNX_CODEGEN_SPEC WP-O3).

---

## 1. WP-G1: understand + ship the vl=8 win (first — it reorders everything else)

vl=8 vs vl=16 is **-0.31 s** (1.54 → 1.23). Plausible mechanisms, untested:
(1) zmm frequency licensing / downclock on 512-bit FMA; (2) the 14² conv
interiors joining at vl=8 (~5×116M MACs); (3) smaller vl-peel tails
(196 % 8 = 4 vs 196 % 16 = 4 — no; but 54 % 8 = 6 vs 54 % 16 = 6 — also no);
(4) 2× better load-port utilization on ymm.

1. **Spike (attribution):** run `--vl 8` with the conv Stage 1d disabled vs
   enabled (quick hack-edit or an option) to split the -0.31 s between
   "14² convs joined" and "GEMMs prefer ymm". Also `perf stat -e
   core_power_lvl*` or frequency counters for the downclock hypothesis.
2. **Decide the default.** If GEMMs genuinely prefer ymm on this machine, the
   machine-model (Analysis/MachineModel) should carry it: a `preferredVL`
   per-band cost decision (small spatial extent → smaller VL; see WP-G3),
   not a hardcoded 8.
3. Re-run the full 4-config table; commit the new default + the attribution
   numbers in the message.

Acceptance: bench table reproduced with the chosen default beating o3;
attribution sentence for the 0.31 s in the commit message. ~½ day.

## 2. WP-G2: the 15 jammed-scalar GEMMs with DRAM accumulators (biggest known leftover)

Symptom (vl=16 census): every N=196/49 1x1 GEMM but one ends as

```mlir
affine.for %oc = 0 to 1024 step 8 {       // mr-jammed
  affine.for %j = 0 to 196 {              // NOT vl-stepped
    affine.for %k = 0 to 256 {            // 8 unrolled replicas inside:
      %c = affine.load %acc[%n, %oc, %j]  //   DRAM load
      ... mulf/addf ...
      affine.store ..., %acc[%n, %oc, %j] //   DRAM store, per k-iter!
```

Stage 3 jammed but neither vectorized nor promoted the accumulators — the
worst possible form (the demote-leftover pathology, x8). Two independent
fixes, both worth doing:

1. **Diagnose the vectorizeBroadcastBand bail (root cause).** One N=196 band
   DID vectorize (`0 to 192 step 16` + tail), 15 didn't, same apparent shape
   (acc `[n, oc', j]` stride-1 in j; weight `[oc', k]`; input `[n, k, j]`
   stride-1 in j → Broadcast). Suspects: detectFamily axis pick on the
   transposed weight (memory says "16 mis-classify Dot"), the post-jam
   re-find ordering, or an alias/perfect-body guard. Spike: extract ONE
   failing band into a lit-size case, run with debug output, find the bail
   point. Fix THAT (likely small), not symptoms.
2. **Promote v2: multi-accumulator bands (safety net).** The promote pass
   currently requires `accs.size() == 1`; the jammed body has 8 independent
   same-shape accumulators. Generalize matchBand/promoteBand to N
   accumulators (one iter_arg each, same band rebuild — Stage 3's own
   `promoteReductions` is the single-loop template). Then even when a future
   band slips through vectorization, it degrades to registers, not DRAM.

Estimated stake: 15 GEMMs ≈ 29 of 53 demoted contractions' FLOPs at N=196/49
(1024·256·196 ≈ 51M MACs each tier ×~13) ≈ 0.6–0.9 GFLOPs — at the measured
scalar ~4 GFLOP/s that's ~0.2 s, and these are also the bands currently
paying the 8×-DRAM-accumulator tax, so the real recovery may be larger.
Acceptance: census shows ≥30/33 GEMMs vectorized (or promoted if k<VL),
bench moves, lit green. ~1–1.5 days.

## 3. WP-G3: per-band VL selection (subsumes "sub-VL convs")

After G1, VL is a per-band decision, not a pass option: a band whose
vectorizable extent is W should get the largest VL in {16, 8, 4} with
`interiorWidth >= VL` (and the machine-model ymm/zmm preference from G1).

- Covers the **2× 7² convs** (interior 4 → VL=4) and any future small-N
  GEMM (N=49: 49 % 16 = 1 vs main 48 at VL=8/4 — already fine).
- Implementation: thread a `pickVL(extent)` through Stage 1d and the
  Stage 3 broadcast path; the vectorizers already take VL as an argument,
  so this is plumbing + a 10-line policy, NOT new kernels.
- Do NOT add VL=4 paths speculatively elsewhere; only where the census
  shows a sub-VL extent.

Acceptance: 7² convs show `step 4` vector bands; no regression at 56²/28².
Half a day after G1. (Stake is small — 2 convs ≈ 230M MACs ≈ 0.05 s — do it
for completeness, don't expect a headline.)

## 4. WP-G4: eltwise/BN fusion (= ONNX_CODEGEN_SPEC WP-O3, now the likely ceiling)

Once contractions are ~fully vectorized, the floor is the ~51 separate
eltwise nests (BN scale/shift, ReLU, residual adds) each round-tripping a
>L2 tensor through DRAM, plus their 204 allocs. o3's krnl path fuses some of
these; ours fuses none.

1. **Measure first:** sum of eltwise nest time = (codegen time) − (codegen
   with eltwise nests deleted — build a hacked IR with the nests stubbed to
   estimate the bound, correctness ignored for the probe). If the bound is
   < 0.1 s, STOP — don't build a fusion pass for noise.
2. If it's worth it: fuse producer→consumer eltwise nests (same shape, same
   batch loop) with the existing affine-loop-fusion cost-model machinery
   (drcomp-fuse exists; it was built for exactly this shape of decision),
   or a dedicated greedy same-shape eltwise merger. Epilogue-fusion INTO the
   conv/GEMM vector kernels (ReLU on the final vector before the store) is
   the v2 of this — bigger win, more invasive.

Acceptance: bench moves or the measured bound says stop. 1–3 days depending
on verdict.

## 5. Explicitly OUT (this round)

- Batch > 1, other models (mnist covered by existing lit/intermediates;
  generalize only when resnet50 is settled).
- im2col / Winograd / packed-GEMM rewrites — different campaign.
- Stride-2 convs + 7x7 stem vectorization (stake ~0.1 s, needs a gather or
  a stride-2 vector kernel — revisit only if G1–G4 leave us behind o3).
- detectFamily refactor beyond the G2 root-cause fix.
- Threading/parallelism (onnx-mlir o3 here is also single-thread; changing
  that changes the comparison for both sides).

## 6. Order + the standing gates

**G1 → G2 → G3 → G4**, bench table after each, all four configs, same
session, back-to-back (thermal fairness). Every WP:

- `scripts/onnx-codegen-bench.sh test/ONNX/resnet50-v2-7.onnx` exits 0,
  norm-rel-err ≤ 1e-4, top-1 858 unchanged.
- Full resnet50 dr-opt run exits 0 (the §0a lesson from 3b: lit alone does
  NOT catch real-IR breakage; there is still no whole-model lit case).
- 1x1 broadcast count never drops below the current 142 census without an
  explained replacement (e.g. G2 raising it).
- lit suite 215+/0; PolyBench register-block tests byte-identical (the
  GEMM stages are shared with the PolyBench campaign — R1 from 3b).
- One commit per step: `onnx_codegen_g1a`, `g1b`, … NEVER push.

## 7. Mechanics (carry-overs that bite)

- Bench: `./scripts/onnx-codegen-bench.sh test/ONNX/resnet50-v2-7.onnx
  [--vl N] [--configs ...] [--workdir DIR]`; workdir keeps every
  intermediate `.mlir` for census greps.
- Image `onnx-mlir-opt --convert-krnl-to-llvm` has NO `affine.vector_load`
  pattern — affine must be lowered host-side first (the script does this).
- Host dr-opt build: `ninja -C build dr-opt`; lit: `/usr/bin/lit -s
  build/test/`. Lowering for kernel spikes: marco LLVM 22
  (`/home/tor/Dev/marco/install/llvm-project/bin`).
- lean-ctx mangles multiline/docker shell commands — write a script file and
  run it.
- Timing: never trust a single run; medians, back-to-back configs, and
  FLOP-count before believing a GFLOP/s claim.
