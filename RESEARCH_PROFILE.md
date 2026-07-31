# Research Profile — Tor Andre Haugdahl

A self-contained briefing to hand to a fresh assistant session (e.g. Claude Cowork).
Written 2026-07-29 from the `drcompiler` repo (`paper-eval` branch), the design
book in `docs/design/`, the campaign reports in `docs/` and `claude-docs/`, and
the paper drafts under `~/PhD/Papers/`.

---

## 1. Who I am

PhD researcher in compiler construction. Primary expertise: **MLIR / LLVM
out-of-tree pass development** — SSA and dataflow analysis, the affine dialect,
interprocedural provenance analysis, analytical cost models, and lit-based
compiler testing. I write the passes, the cost models, the benchmark harnesses,
and the papers.

Secondary track (co-authored, not compiler work): embedded / cyber-secure
wearable sensor systems (CAPSARII), which appears in my publication record but
is not what I use an AI assistant for.

Working language for code and papers is English; my shell locale is Norwegian
(`LC_ALL=C` is forced in benchmark scripts because a comma decimal separator
breaks `printf`/`awk`).

---

## 2. The artifact: `drcompiler`

An out-of-tree MLIR compiler, ~5k+ LOC of pass code plus a large test and
benchmark corpus. It has **two jobs sharing one foundation**:

1. **Cache-aware data recomputation** — decide, per `memref.load`, whether to
   re-derive a value from its producers instead of paying a memory round-trip.
2. **Machine-model-driven codegen** — turn the naïve affine loop nests a tensor
   compiler emits (onnx-mlir, Polygeist) into register-blocked, vectorized,
   cache-tiled, and now *parallel* kernels.

Both are decisions about the same scarce resources (cache capacity, memory
bandwidth, vector registers, FMA pipes), so both consult one description of the
machine.

### The spine: `MachineModel`

`drcompiler::MachineModel` is the single resolved description of the target
(cache geometry, latencies, `llcSharers`, page/TLB, vector-exec model,
`vecRegBudget`, throttle). Resolution order: `defaults < JSON < explicit CLI`.
The standing rule: **no pass carries a private copy of a machine parameter**;
hard-coded knobs are the bug the Machine Model exists to remove. Cost is priced
through one primitive, `dr::estimateLoadLatency` (bytes → L1/L2·occ/effL3/mem
latency tiers), so contention-awareness (`effL3 = l3Size / llcSharers`) reaches
every consumer for free.

### Pass inventory

- **Recomputation family**: `raise-malloc-to-memref`, `data-recomputation`
  (interprocedural load/store provenance → SINGLE / MULTI / LEAKED / KILLED,
  optional recompute + partial rematerialization), `memory-fission` (inverse of
  loop fusion: split fused siblings and materialize a shared expensive
  subexpression when the cache model says it pays).
- **Codegen family**: `affine-register-block` (micro-kernel `mr × ⌈nr/vl⌉`
  vector accumulators, macro-tile alignment, cache tiling, `gemmBlocking`),
  `dr-scalar-reduction-demote` / `-promote`, `canonicalizeAllocaGemm`,
  vectorizer with per-band family detection, `dr-affine-loop-tile` (LLC-gated),
  `affine-stencil-time-tile`, forked loop-tiling / -fusion / -distribution.
- **Parallel family**: `dr-par-bubbles` (+ `par-spmd` mode), the `par` dialect,
  `ParAliasOracle`, `convert-par-to-omp`, barrier elision, `par.critical` /
  `par.reduce` / dynamic-extent shards.

### Pipeline

```
source.c → cgeist (Polygeist, LLVM 18) → .mlir → [DLTI sed fixup]
  → dr-opt (LLVM 22) → mlir-opt lowering → mlir-translate → clang -O3 -march=native
```

Front end and optimizer are **deliberately decoupled** — no shared library, only
textual `.mlir`, so the optimizer tracks upstream LLVM independently. The ONNX
path substitutes onnx-mlir `--EmitMLIR` for cgeist. `drcc` is the drop-in
`cc`/`c++` wrapper for SPEC CPU 2017.

### Invariants I hold

- **Byte-identical by default.** Every capability added after the original model
  (compute/bandwidth roofline arms, thread model, `gemmBlocking`, `poly-approx`,
  `canonicalizeAllocaGemm`) is latched behind a `hasExplicit*` flag or a pass
  option and is inert until opted in. The lit suite proves it against golden IR.
- **Correctness gate on every perf claim**: `norm-rel-err ≤ 1e-4` vs the
  untransformed reference, plus a checksum; for parallel work, byte-identical
  golden output across thread counts.
- Lit suite currently ~240 tests, kept green.

---

## 3. Research narrative — where it has been and where it is

Track it as a sequence of measured campaigns, each with a written findings doc.

1. **Data recomputation (original thesis direction).** The analysis is real and
   works. The *speedup* story did not clear the bar: PolyBench is a structural
   mismatch (hot loads are read-only inputs; recomputing a reduction always
   loses — see `research/NO_POLYBENCH.md`), and three spikes (E, H-Ryzen,
   H-Xeon) reached honest NO-GO verdicts in May 2026. I lost faith in data
   recomputation as the *downstream consumer* of the analysis and pivoted.
2. **Cost model v1 → v4.** Unification campaign: `MachineModel` as
   single-source-of-truth, contention-aware reuse distance (effective LLC =
   L3/sharers), portability and robustness audits. `CROSSCUTTING.md` is the
   honest 5-lens audit of how unified it actually is — it names two
   non-communicating register models, three disconnected notions of vector
   width, two contention-blind cache forks, and dead canonical helpers. No
   roofline/bandwidth term existed for parallelism; that gap is specified there.
3. **Codegen campaign.** Register blocking generalizes across BLAS-3 and tensor
   contractions (2.3–3.4×) via a vector-dialect broadcast micro-kernel that
   cracks the LLVM-SLP-on->2D wall. Operand packing was measured and is a red
   herring — do not build it.
4. **ONNX / transformer campaign.** Headline discipline matters here: an early
   "1.6× beats onnx-mlir `--O3`" was an SSE2-baseline artifact and was
   **retracted**. Backend-matched re-measure on openai-gpt (Zen4): our
   `codegen` 0.231 s vs `o3host` 0.367 s — **1.59× pure-transform win**; but
   onnx-mlir's native `--EmitObj` back end (0.224 s) edges us on the net number,
   so the remaining gap is *lowering*, not transform. Biggest shared headroom is
   scalar transcendentals (Gelu `powf(x,3)`+tanh, 20–45 % of runtime).
5. **Parallel / SPMD campaign (most recent).** Whole-function SPMD on real
   models: resnet50 batch-1 materializes and is numerically exact end-to-end
   through onnx-mlir; we beat `onnx-mlir --parallel` on both resnet50 (9.5×) and
   openai-gpt (1.5×, 2.8× at batch>1) via a vec×par no-cache-tile recipe. Full
   PolyBench parallel sweep with measured OpenMP scaling (gemm 14.4×, jacobi-2d
   18.8×–20.6×, and honest regressions: atax 0.52×). Barrier elision is sound
   and byte-identical but perf-neutral — the cap is Amdahl (critical bands), not
   barriers, and that is reported as such.

**Repeating meta-finding:** the cost model is what makes results portable and
publishable; the transforms are only as trustworthy as the model they were
derived from, and negative results are load-bearing evidence, not failures.

---

## 4. Papers

- `~/PhD/Papers/data-recomputation` — *"A Cache-Aware Static Analysis of Load
  Provenance for Memory-Materialisation Decisions in MLIR"*. Has a written
  self-assessment (`CGO_FEASIBILITY.md`): honest verdict was **not CGO-worthy as
  of May 2026** — strong CC/PACT, borderline CGO; needs more empirical evidence
  or a sharper theoretical contribution.
- `~/PhD/Papers/affine-machine-model-opt` — *"Crafting Machine Models for MLIR
  Transforms"*. The cost-model-as-contribution line, which is where the current
  evidence is strongest.
- `docs/design/` — a Typst + CETZ design book (9 chapters + appendix) spined on
  the Machine Model, including a case-study chapter on closing the openai-gpt
  gap and a decisions chapter that documents the *measured negatives* alongside
  the positives.
- CAPSARII papers (conference + MicPro journal) — co-authored, embedded sensors,
  separate track.

---

## 5. Method and evaluation infrastructure

**The loop:** hypothesis from a model reading → cheapest possible spike (probe,
not build) → measure with median + correctness gate → win means commit locally,
no-go means revert *and document, because the no-go is data*.

Benchmark suites and harnesses in `scripts/` (~60 scripts):
- PolyBench (campaign generator, per-family sweeps, tiling/spill/roofline
  probes, parallel driver, verification).
- ONNX models via `drcc-benchmarks/onnx/scripts/onnx-run-bench.sh` — mnist,
  resnet50-v2-7, openaigpt, gptneox; configs `none` / `codegen` / `o3host` split
  transform quality from back-end quality so neither confounds the other.
- SPEC CPU 2017 through `drcc`; MARCO/Modelica bridge (`dr-raise-scf-to-affine`).
- Cost-model calibration (`calibrate_weights.py`, `calibrate_regpress.py`,
  Nelder-Mead-style fitting), cache-latency and cross-thread roofline
  microbenchmarks, `gen_cpu_cost_model.py` for machine-model JSON generation
  with double-pump detection.

Measurement hygiene I already learned the hard way: always name the actual
`-march` on *both* sides of a comparison; single-machine AVX2-only results get
discounted by reviewers; noise and `pkill -f` self-matching have burned me;
`pgrep -f` matches the watcher itself.

---

## 6. How I want an assistant to work with me

- **Be ruthlessly honest.** Weak findings get stated as weak, with numbers. No
  "promising direction" softening. A NO-GO reported early is the highest-value
  contribution — it saves a dead campaign.
- **Spike before commit.** Do not propose more code without answering: does the
  data warrant it? No optimistic novelty claims before the underlying speedup
  hypothesis validates.
- **Name the competition.** If Pluto, Polly, Scout, RPG2, onnx-mlir `--O3`, or a
  published CGO/CC paper overlaps my claim, say so and assess the gap.
- **Aim at the venue the data supports**, not above it.
- After a negative result, give ranked pivots with risk levels — not "try the
  same thing harder".
- **Conversation style: terse.** I run caveman mode (`/caveman:caveman`) — drop
  articles and filler, keep every technical term exact, quote errors verbatim.
  **Files, code, commits, and papers stay in normal full prose.**
- Verify against the source before asserting; every claim about the codebase
  should be tied to `file:line`. I check.
- I work locally and control my own pushes — commit when asked, don't push.

---

## 7. Open threads (as of 2026-07-29)

- Cost-model unification gaps from `CROSSCUTTING.md`: reconcile the two register
  models, the three vector-width notions, the two contention-blind cache forks;
  wire up the dead `effectiveLLC()` / `tlbReachBytes()` helpers; add the missing
  bandwidth/roofline currency for parallel workloads.
- Transcendentals (`powf`/tanh in Gelu) — the largest shared headroom vs
  onnx-mlir on transformers.
- The back-end gap: our krnl→affine round-trip lowers ~1.66× slower than
  onnx-mlir's native `EmitObj`, which currently eats a real transform win.
- Parallel SPMD remaining: halo exchange (S4), `libdrpar` pinning (S5),
  reductions (S6); OpenMP execution through onnx-mlir is blocked by
  `convert-krnl-to-llvm` rejecting external `omp`/`scf.parallel`.
- Which paper the current evidence best supports — the machine-model line looks
  stronger than the data-recomputation line.
