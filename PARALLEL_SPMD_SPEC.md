# PARALLEL_SPMD_SPEC.md — Whole-Kernel SPMD Owner-Computes Sharding

Status: design spec (campaign, 2026-06-18). Builds on:
`PARALLEL_BUBBLE_SPEC.md` (front-end widening, landed M0–M4),
`PARALLEL_PAR_DIALECT_SPEC.md` (the `par` dialect),
`PARALLEL_CODEGEN_SPEC.md` (the `dr-shard` / `libdrpar` pinned-pool back-end).

## 0. Thesis

The most aggressive *sound* parallelization the track can target: **fork the
pinned team once per kernel, shard ONE axis through the entire kernel, give each
pinned thread a fixed slice of every buffer, and run the whole kernel as N
near-independent sequential programs — eliding inter-layer barriers wherever the
shard→shard data flow is owner-aligned.**

It is the apex of bubble-widening: instead of widening *adjacent* bands, widen
the **whole function into one maximal `par.region`** over a common shard axis;
`par.barrier` / `par.redistribute` / `par.reduce` survive only at genuine
cross-shard edges.

```
   naive (per-loop fork/join)          whole-kernel SPMD (this spec)
   ┌──────┐ barrier                    fork ONCE
   │ L1   │   │                        ┌───────────────── team ─────────────────┐
   └──────┘   ▼                        │ core0   core1   core2   core3           │
   ┌──────┐ barrier                    │ L1[s0]  L1[s1]  L1[s2]  L1[s3]  (aligned)│
   │ L2   │   │                        │ L2[s0]  L2[s1]  L2[s2]  L2[s3]  (aligned)│
   └──────┘   ▼   …  N barriers        │ ════════ par.barrier (cross-shard) ═════│
   ┌──────┐                            │ L3[s0]  L3[s1]  …                        │
   │ L3   │                            └─────────────────────────────────────────┘
   └──────┘                            join ONCE.  barriers only at true edges.
```

Overhead floor: **1 fork/join total**; barriers only where the shard axis
carries a cross-shard dependence (zero for the pointwise/eltwise/owner-aligned
chains that dominate transformer & CNN runtime).

## 1. The four constraints → how they're met

| Constraint | Mechanism |
|---|---|
| **Minimum overhead** | Fork-once (persistent pinned pool) + **barrier elision** (§4). Inter-layer sync → near-zero. |
| **Conditioned on pinning** | Owner-computes residency: shard `t`'s intermediates live in core `t`'s **private L2** across all layers; producer→consumer hand-off is same-core, cache-hot; first-touch puts buffers NUMA-local. Unpinned ⇒ the residency + elision argument collapses ⇒ gate OFF (fall back to per-region fork/join). |
| **Resistant to context changes** | All static: block partition (no work-stealing / dynamic schedule), shard count chosen **once** for EXCLUSIVE mode, `WS/N` exact. Cost gate uses the **private** L2 (resident regardless of LLC co-tenants). No dependence on scheduler decisions or co-tenant noise. |
| **Sharding** | One contiguous block-partitioned axis threaded through every layer (owner-computes). |

## 2. The model

A **persistent pinned team** (the `libdrpar` pool, `PARALLEL_CODEGEN_SPEC.md`
§2) is forked once at kernel entry. A single **shard axis** `s` is partitioned
into contiguous worker-owned ranges `R_w`; worker `w` executes, for *every*
layer, the iterations of that layer's shard loop with `s ∈ R_w`, and owns the
`R_w`-slice of every intermediate buffer (first-touch placement). The sequential
glue between hot loops (reshapes, scalar setup) runs in a `par.critical`
(single worker) or is replicated when pure+cheap.

**Batch ≥ cores** is the trivial case: shard the batch → zero sync ever, perfect
residency. The hard target is **batch-1 latency**, where `s` is a within-sample
axis (spatial / channel / GEMM-M) and §4 barrier-elision earns the win.

## 3. Shard-axis selection (analysis)

Any choice is sound; the choice decides how many barriers elide. Select a global
shard dimension `d` and, per region, the loop carrying `d`:

- **Candidate axes**: for each region's maximal band, each `parDim`
  (`PARALLEL_BUBBLE_SPEC.md` §1) is a candidate, tagged by the *buffer dimension*
  it indexes on the dominant (largest, layer-flowing) tensor.
- **Score** a global dimension `d` by: (a) number of regions where `d` is a
  `parDim` (more ⇒ fewer reduction/transpose breaks), weighted by region cost;
  (b) `WS/cores` along `d` fits the **private** L2 (`MachineModel::
  effectivePrivateCache`); (c) contiguous / unit-stride (block partition ⇒
  spatial locality + clean first-touch).
- Pick the max-scoring `d`. Regions where `d` is **not** a `parDim` (a reduction
  over `d`) keep `d` sequential within the shard, or trigger §6.

Transformers → token/M axis (parallel across eltwise + GEMM-M). CNNs →
batch×out-channel or spatial rows.

## 4. Barrier elision — the core analysis (the S1 spike)

Between consecutive regions `B_k`, `B_{k+1}` both sharded on `s`, a barrier is
needed **iff** some memory location written in `B_k` is read/written in `B_{k+1}`
by a **different** worker. Project every cross-region dependence onto `s`
(extend the M3 machinery: `alignedAccess` / `classifyCross` / `findReshuffle`,
`lib/Transforms/ParBubbles.cpp`), evaluated on the shard axis and the worker
owner-ranges:

| shard-axis relation of the dependence | verdict |
|---|---|
| **aligned** — same `s`-component map (shard `t` reads exactly what shard `t` wrote) | **ELIDE** (no barrier) |
| **bounded offset** `K`, `\|K\| ≪ shard width` (conv/stencil halo) | **HALO** — elide the full barrier; handle the `K`-boundary by redundant recompute (§4.1) or thin exchange |
| **coefficient change** (reverse / permute / transpose) | **REDISTRIBUTE** (`par.redistribute`, already detected) |
| **reduction over `s`** | **REDUCE** (`par.reduce`, §6) — one barrier + tree combine |
| anything unprovable | **BARRIER** (conservative) |

**Soundness (elision legality).** Let `a∈B_k` write location `L_a(s)` at
iteration `s` and `b∈B_{k+1}` access `L_b(s')`. Owner-computes ⇒ `owner(a)=w`
where `s∈R_w`, `owner(b)=w'` where `s'∈R_{w'}`. Eliding the barrier is legal iff
`∀` dependences `(a,b)` (≥1 write): `L_a(s)=L_b(s') ⇒ owner(s)=owner(s')`. The
**aligned** case (`L_a`,`L_b` the same affine function of `s`, coefficient ±1,
equal constant on the same buffer) gives `L_a(s)=L_b(s') ⇒ s=s' ⇒ owner equal` —
sound. Everything not provably aligned keeps its sync (HALO/REDISTRIBUTE/REDUCE/
BARRIER). No speculation; static affine model only (`PARALLEL_BUBBLE_SPEC.md` §2).

### 4.1 Halo (bounded offset) — redundant recompute

For an offset `K` with `|K| ≪ shard width`, only the `K` boundary iterations of
each shard cross to a neighbor. The aggressive, sync-free resolution: each worker
**redundantly recomputes** its `K`-element halo from inputs it already owns
(ghost zones), trading `O(K)` compute per shard for **zero** cross-thread
traffic. Gate on the cost model: redundant-compute cost `< barrier cost` (§5).
Fall back to a barrier when the halo producer is itself expensive.

## 5. Cost-model integration

Reuse `decideShard()` (`PARALLEL_CODEGEN_SPEC.md` §6) + the `ThreadModel` spawn /
barrier terms proposed in `PARALLEL_BUBBLE_SPEC.md` §7 (`spawnCycles`,
`barrierCycles`, default 0 ⇒ inert). Add:

- **Private-L2 residency gate**: choose shard count `cores` so the per-shard
  working set of the *whole chain* fits `effectivePrivateCache(L2)` at the
  configured occupancy (`l2-occupancy-pct`) — the contention-robust budget
  (resident regardless of LLC co-tenants). This is what makes the opt
  context-resistant.
- **Elision payoff**: model the win as `barriers_elided × barrierCycles` saved.
- **Halo tradeoff**: redundant-recompute iff `haloIters × perIterCompute <
  barrierCycles`.
- **Mode**: EXCLUSIVE (own the machine) is the target; `cores` may be `<
  activeThreads` in INTERSPERSED.

Default machine (no explicit thread model) ⇒ `cores == 1` ⇒ nothing emitted ⇒
**byte-identical**, consistent with every drcompiler pass.

## 6. Reductions over the shard axis

When `d` is a reduction in some region (GEMV, softmax-sum, norm with no parallel
outer): emit `par.reduce` (`PARALLEL_PAR_DIALECT_SPEC.md` §3.6) — per-worker
padded partials + tree combine, one barrier, under `fastmath<reassoc>` /
`--par-deterministic`. This is the irreducible sync floor; prefer a shard axis
where `d` is parallel in the hot regions so this is rare.

## 7. Materialization & lowering

The SPMD form reuses the existing dialect — no new ops required:

```
par.region {                         // one persistent team for the whole kernel
  par.forall (%s) in (shard space) { // hoisted to the common shard axis
    <layer 1 body for shard s>
    <layer 2 body for shard s>       // no par.barrier here  => ELIDE
    par.barrier                      // only at a true cross-shard edge
    <layer 3 body for shard s>
    par.yield
  }
}
```

Whole-function widening hoists one `par.forall` over `s`, sinking every layer's
shard loop into it (inner non-shard loops stay sequential / `scf.for`).

**Lowerings:**
- `par → omp` (**new, faithful**): `par.region → omp.parallel` (one team);
  `par.forall → omp.wsloop` (static schedule); an **elided** boundary →
  `nowait` on the preceding `wsloop` (no implicit barrier); `par.barrier →
  omp.barrier`; `par.reduce → reduction`. This is strictly more efficient than
  the existing `par → scf.parallel → --convert-scf-to-openmp` compose, which
  fork/joins per loop.
- `par → libdrpar` (Phase B): outline the whole `par.forall` body once into
  `__drpar_for(...)`; each sunk shard loop iterates the worker's `[lo,hi)`;
  `par.barrier → __drpar_barrier`; first-touch owner-computes placement via
  `__drpar_alloc_local`. The pinning/residency win.
- `par → scf` (existing): sequential reference for correctness diffs.

## 8. Milestones (spike-first: measure elision before building the backend)

| S | Deliverable | Gate (measurable) |
|---|---|---|
| **S0** | Shard-axis selection (analysis + remark): chosen `d`, #regions parallel on it. No mutation. | On resnet50 + a transformer, the chosen axis is a `parDim` in the majority of hot regions. |
| **S1** | **Barrier-elision analysis** (the spike): per inter-region edge → ELIDE/HALO/REDISTRIBUTE/REDUCE/BARRIER, with counts (diagnostic only). | **Measure barriers eliminated vs naive per-loop** on real kernels — the payoff number, *before* any backend work. Go / no-go. |
| **S2** | Whole-function widening + materialization (one `par.region`, hoisted `par.forall`, `par.barrier` only at non-elided edges). | `par → scf` byte-identical to the sequential kernel; IR barrier count == S1. |
| **S3** | Faithful `par → omp` (one team + `nowait`). | End-to-end run correct; fewer fork/joins than the scf-compose path; measured speedup vs sequential. |
| **S4** | Halo redundant-recompute (§4.1). | A conv/stencil chain: halo barrier eliminated, correct (`norm-rel-err ≤ 1e-4`), faster than the barrier version. |
| **S5** | `par → libdrpar` (pinned pool + first-touch owner-computes). | Pinned EXCLUSIVE-mode latency vs omp; per-core private-L2 residency measured (the X3D V-cache / NUMA story, `PARALLEL_CODEGEN_SPEC.md` §5). |
| **S6** | Reduction-over-shard-axis (`par.reduce`). | GEMV / softmax-sum cross-thread combine correct. |

## 9. Soundness & defaults

- Static affine model only (distinct-root non-aliasing, the affine ecosystem's
  contract); barrier elision is the legality theorem of §4. No runtime
  speculation / versioning (the unsound frontier — explicitly OUT).
- **Pinning is a precondition**: without an explicit pinned topology the
  residency + elision argument is void ⇒ the opt gates OFF (per-region fork/join
  via the existing path).
- Default OFF / byte-identical until a thread model + pinning are configured.
- Methodology (per the track): spike-first (S1 measures payoff before the
  backend), one commit per step, honest go/no-go, NEVER push.

## 10. Tests

- S0/S1: `-verify-diagnostics` remarks — chosen shard axis; per-edge elision
  verdict + elided/kept counts on hand-built multi-layer kernels (eltwise chain
  → all ELIDE; conv chain → HALO; transpose → REDISTRIBUTE; GEMV → REDUCE).
- S2: FileCheck — one `par.region`, one hoisted `par.forall`, `par.barrier` only
  at the seeded cross-shard edge; `par → scf` numeric equivalence.
- S3: FileCheck — `omp.parallel` with `omp.wsloop ... nowait` at elided edges,
  `omp.barrier` only where kept.
- S4: execution diff (redundant-halo vs barrier vs sequential).

## 11.5 Spike results (S0/S1) — measured 2026-06-18 (commit 59c1dcc)

S0/S1 landed diagnostic-only (`dr-par-bubbles{par-test-spmd}`). Measured:

- **Synthetic clean-affine eltwise chain** (3 loops, one axis): **2/2 edges
  ELIDE**. (Note: M3 already *fuses* this case into one `par.forall` — strictly
  stronger than elision. The unique S2+ value is the multi-dim non-fusable case.)
- **resnet50 affine dump** (real; needs `-allow-unregistered-dialect` for the
  `krnl.*` ops): over the **batch (`dyn`) axis**, 117/370 parallel bands chain
  with **33/33 edges ELIDE, 0 halo / redistribute / barrier** → whole-kernel
  SPMD over batch is **completely barrier-free**. 676/844 loops classify
  PARALLEL (93 reduction, 72 conservative, 3 carried).

**Verdict: GO on the mechanism** — barrier elision is sound, works on a real
conv net, and confirms batch≥cores is embarrassingly parallel, end-to-end
barrier-free.

### Blockers for S2–S6 (why the build-out needs supervision / more infra)

1. **krnl dialect**: onnx-mlir affine dumps still carry `krnl.global` etc.;
   `dr-opt` can't parse them without `-allow-unregistered-dialect`, and as
   opaque ops they would block real materialization. Need to run at a pipeline
   stage without krnl, or register/handle it.
2. **Body complexity**: 72/844 loops are conservative — per-layer `memref.alloc`
   scratch + non-affine `memref.load` + `arith.select` make the affine-only
   oracle bail. Real materialization must handle (or pre-clean) these.
3. **Shard-axis at batch-1**: the by-coverage heuristic picks the batch axis; at
   batch=1 (latency) that is degenerate (1 shard). Within-sample axes
   (channel/spatial) don't chain across layers (64→128→256), so batch-1 latency
   has no single whole-kernel shard axis — it needs per-segment re-sharding +
   redistributes (the hard case).
4. **Validation**: S2 elision soundness *cannot* be validated by `par→scf`
   (which joins between `scf.parallel`s anyway); it needs `par→omp(nowait)` or
   libdrpar **plus execution**. S4 (halo recompute) and S6 (reduce reassoc) are
   numeric; S5 (libdrpar) needs a pthread-pool C runtime that does not exist.
   None are execution-validatable unsupervised → **not built** (soundness bar).

### Recommended supervised next step
S2 on a **clean, batch>1** affine kernel (no krnl; scratch allocs cleaned):
hoist the batch loop to wrap the function as one `par.region`/`par.forall`,
place `par.barrier` only at non-elided edges, lower via a faithful `par→omp`
(one team + `nowait`), and validate by **execution** (numeric diff vs
sequential). Then libdrpar for pinning. The payoff (33/33 barrier-free on
resnet50 batch) justifies it.

## 11.6 S2 landed — structural tier (2026-06-19, commit e1ae9eb)

S2 whole-function widening + materialization, behind `dr-par-bubbles{par-spmd}`
(IR-mutating, default off / byte-identical). Builds the §7 form on clean affine
kernels: ONE `par.region`; maximal ELIDE-connected shard-band runs share ONE
hoisted `par.forall` over the (constant) shard axis (owner-computes — bodies
sequenced, inner loops sunk as `scf.for`); a non-elided edge → `par.redistribute`
(1-D remap) or `par.barrier`. IR sync-op count == S1's non-elided-edge count.

- **Demonstrated** (tests `spmd-materialize.mlir`): eltwise chain (3 bands, all
  ELIDE) → one forall / zero barriers; **multi-dim batch shard** — two
  owner-aligned bands fuse into one `par.forall(n)`, a transpose-read third band
  (genuine cross-shard) behind a `par.barrier`, inner spatial loop as `scf.for`.
  This is the multi-dim non-fusable case M3 cannot express (M3 fuses only
  conformant depth-1 siblings). `par→scf` gives the sequential reference.
- **Sound bails** (no mutation + naming remark): dynamic shard extent, off-axis
  / non-materializable band, non-contiguous bands, inter-band value dependence —
  exactly the §11.5 real-kernel blockers (resnet50-style dumps correctly bail).

**Tier when landed: structural.** S3 (below) lifted it to execution-validated
for clean kernels.

## 11.7 S3 landed — execution-validated (2026-06-19, commit ce60d2f)

The faithful `par→omp` lowering (`convert-par-to-omp`) + the execution gate that
was §11.5 blocker #4.

- `par.region → omp.parallel` (one team); `par.forall → omp.wsloop {
  omp.loop_nest }`; an elided edge → `nowait` on the wsloop (the explicit
  boundary `omp.barrier` provides the sync — no double barrier); `par.barrier`
  / `par.redistribute → omp.barrier`. One `omp.parallel` + barriers only where
  S1 kept them — strictly fewer fork/joins than the per-loop
  `par→scf.parallel→convert-scf-to-openmp` compose. Test `par-to-omp.mlir`.
- **EXECUTION-VALIDATED** (`scripts/validate-spmd-omp.sh`): a clean multi-dim
  shard kernel (two owner-aligned ELIDE bands fused into one `omp.wsloop nowait`
  + a transpose band behind the one kept `omp.barrier`) compiled three ways —
  golden (untransformed, sequential), seq (`par→scf`), omp (`par→omp →
  convert-openmp-to-llvm`, real OpenMP team via `mlir-runner`). All three
  **byte-identical**; 30 omp runs across {1,2,3,4,8,16} threads, **zero
  mismatches**. → the barrier elision + `nowait` placement is execution-correct,
  not just structurally plausible. **§11.5 blocker #4 cleared for clean
  kernels.**

## 11.8 Measured speedup — S3 fully met (2026-06-19, commit 8320452)

The perf gate (`scripts/gen_spmd_perf_kernel.py` + `scripts/bench-spmd-omp.sh`):
a large clean batch-shardable kernel — K owner-aligned elementwise layers over
batch N, each element a P-step compute recurrence (compute-bound). All K layers
fuse into ONE `par.forall` over N with **ZERO barriers** (the barrier-free batch
SPMD case). golden-sequential vs `par→omp` across threads, checksum-checked:

| config | per-call | speedup | checksum |
|---|---|---|---|
| golden seq | 0.854 s | 1.00× | ref |
| par→omp 1t | 0.891 s | 0.96× | OK (lowering overhead ~4%) |
| par→omp 4t | 0.225 s | 3.80× | OK |
| par→omp 8t | 0.114 s | 7.50× | OK |
| par→omp 16t | 0.058 s | **14.69×** (92% eff) | OK |
| par→omp 32t | 0.032 s | ~25× (~78% eff) | OK |

(dev host, 32 cores; N=64 M=8192 K=4 P=384.) Output byte-identical to the
sequential golden every run. → whole-kernel barrier-free batch SPMD is **correct
AND scales near-linearly**. **S3 fully met** (faithful `par→omp`,
execution-correct, measured speedup).

**Honest limits:** compute-bound kernel — a bandwidth-bound kernel saturates
earlier (a hardware ceiling, not an SPMD flaw); 32t falls to ~78% (full-socket
shared-LLC/turbo/SMT). Clean synthetic kernel (no krnl, scratch hoisted); real
ONNX kernels still need the §11.5 blockers (krnl, per-layer allocs, batch-1
axis) before this runs on resnet50. S4 (halo recompute), S5 (libdrpar pinning),
S6 (reduce-over-shard) are unchanged.

## 11.9 Toward real kernels — two §11.5 blockers cleared (2026-06-19)

Investigating the resnet50 `.03-dr` dump grounded the residual gap: weights
(`krnl.global`) all dominate at the function top, **zero deallocs**, and the
inter-band glue is exactly scratch `memref.alloc` + index/view ops. So two
mechanical blockers, both now removed and execution-validated on synthetic
real-ONNX-shaped kernels:

- **Inter-band glue hoisting** (blocker #2, commit d16adbe): a shard-band run
  separated only by hoistable glue (side-effect-free ops or fresh allocs, no
  band-output dependence) now materializes — the glue is hoisted above the
  first band (sound: nothing hoisted reads band-written memory or depends on a
  band). Validated: interleaved-alloc kernel, par→omp checksum-identical,
  14.68× @ 16t.
- **Dynamic (runtime) shard extent** (blocker #3, commit 0417922): `par.forall`
  now carries an optional dynamic upper bound (`dyn(%N)`); S2 shards a `0 to %N`
  batch axis, requiring all bands to share the SAME runtime Value. Lowers to
  `scf.parallel`/`omp.loop_nest ... to (%N)`. Validated: runtime batch N=48,
  par→omp byte-identical to sequential, 15 runs × {1,2,4,8,16} threads, 0
  mismatches (`scripts/validate-spmd-dyn.sh`).

**S2 now expresses the real-ONNX batch-throughput shape** (interleaved scratch
allocs + runtime batch).

## 11.10 Batch-1 within-sample SPMD — resnet50 MATERIALIZES (2026-06-19)

The batch-1 latency path (user-directed).  At batch=1 the batch axis is
degenerate, so each layer shards a WITHIN-SAMPLE axis (its own output:
oc/spatial/output-neuron); the inner reduction (ic/kernel/k) stays within-shard
— no cross-shard reduce for conv/pool/FC.  `par.reduce` (commit 6ecb336) built
for the genuine reductions (GPT softmax).

- **`par-spmd-perband` + whole-function widening**: ONE `par.region` over the
  whole function; each band → `par.forall` (own output axis, dynamic extent ok)
  or `par.critical` (non-shardable, single worker); allocs + pure metadata
  **hoisted** (shared); read-only glue **replicated** (every worker recomputes,
  SSA visible); write/free glue → `par.critical`; `par.barrier` between bands.
- **onnx-mlir-lean image built** here (the krnl→llvm back-half + `libcruntime`);
  harness proven live (mnist codegen 6.5×, 2e-7).
- **Static batch-1 is the unlock**: dynamic-batch (`?`) resnet50 computes buffer
  sizes from *loaded* shape values → data-dependent allocs the widener soundly
  refuses (bails).  Compiling with `--shapeInformation=0:1x3x224x224` (the
  latency case) drops that glue.  `scripts/spmd-resnet50-static.sh`.
- **RESULT** (static batch-1 resnet50, 137 top-level bands): materialized
  **foralls=82, critical=55, moved=207, barriers=136** (82 parallel = 60%); and
  `convert-par-to-omp` lowers it to **ONE `omp.parallel`** with `omp.wsloop=82`,
  `omp.single=55`, `omp.barrier=136`, **par leftover=0** — the whole-kernel SPMD
  team form on a real net, structurally valid end-to-end from affine.

**NUMERIC CORRECTNESS — VALIDATED** (`scripts/validate-spmd-onnx.sh`):
the SPMD-materialized model run end-to-end through the onnx-mlir toolchain is
**byte-identical** to the untransformed reference —
- **mnist**: `norm_rel_err = 0.000e+00` (n=10);
- **resnet50-v2-7 batch-1**: `norm_rel_err = 0.000e+00` (n=1000), exercising the
  full widened structure (82 forall + 55 critical + 207 hoisted + 136 barriers).
The whole-function SPMD transform is numerically exact on a real conv net.

**Parallel (OpenMP) execution — blocked by onnx-mlir's lowering, not the
transform.** `onnx-mlir-opt --convert-krnl-to-llvm` rejects any external
parallel construct (`failed to legalize omp.* / scf.parallel` — it expects
sequential `scf.for`; onnx-mlir parallelizes via its own `--parallel`).  So the
correctness gate lowers `par.forall` to **sequential** cf (par→scf →
`--convert-scf-to-cf`) before krnl-to-llvm.  Actual parallel execution needs a
**krnl-free** lowering path (lower `krnl.global` → `llvm.mlir.global` without
onnx-mlir's monolithic pass, then host `--convert-openmp-to-llvm` + libomp), or
to emit through onnx-mlir's native parallel path — a separate integration.
The mechanism + materialization are proven; only the parallel back-end remains.

## 11.11 Parallel back-end + batch-1 speedup (2026-06-19)

`lower-krnl-global` (commit f751727) routes around onnx-mlir's krnl-to-llvm:
`krnl.global → memref.global`, `krnl.entry_point` erased, then HOST mlir-opt
lowers the whole module to LLVM with omp intact (`--convert-openmp-to-llvm`),
`mlir-translate` emits real `__kmpc_fork`, `clang -fopenmp` links libomp, entry
via a hand-written `_mlir_ciface_main_graph` harness.
`scripts/validate-spmd-parallel.sh`.

Per-band shard-axis fix (commit 6cbfb72): shard the outermost parallel loop with
**real extent** (skip the degenerate batch axis at batch=1; the conv
output-channel / spatial axes, 64–2048).

**resnet50 batch-1, parallel, correct** (`norm_rel_err ≤ 1e-6` at every thread
count): with `dr-affine-loop-distribute` + `dr-scalar-reduction-demote`
perfecting the conv reduction bands (82→148 forall, 55→22 critical, 87%
parallel), the OpenMP run scales **1.00 / 1.18 / 1.28 / 1.33 / 1.36×** at
1/2/4/8/16 threads.

**Honest limit.** The speedup is modest and roughly **break-even vs plain
sequential**: spmd-1t = 3.31 s > plain-seq ≈ 2.2 s, because (a) `demote` ran
*without* the `register-block` + `promote` that normally follow it (slow
memref-accumulator reductions), (b) **169 per-layer barriers**, (c) the convs
are memory-bound.  Batch-1 latency is the inherent hard case (limited
within-sample parallelism + barrier-heavy); **batch ≥ cores THROUGHPUT is where
SPMD wins** — the synthetic near-linear result (14.69× @ 16t, §11.8).  Closing
the batch-1 gap (full codegen passes for a fair baseline, barrier elision
between same-axis layers, or batch > 1) is perf-tuning on top of a proven,
correct mechanism.

**Batch > 1 throughput on resnet50 — also bandwidth-limited.** Sharding the
batch axis (consistent owner-computes across all layers) at batch=16 scales only
**1.00 / 1.36 / 1.39 / 1.36× at 1/8/16/32 threads** (correct, ≤1e-6), plateauing
at 8 threads.  16-way data parallelism yielding ~1.4× means the bottleneck is
**memory bandwidth, not the SPMD mechanism**: resnet50 inference is
memory-bound (large activations, low arithmetic intensity), so parallel cores
saturate DRAM bandwidth.  The contrast is the evidence — the *compute-bound*
synthetic kernel scaled **14.69× @ 16t** (§11.8) with the identical machinery.
(Compounded here by the slow demoted-scalar conv path — no register-block /
vectorize — and 22 serial layers.)

**Net verdict.** The whole-kernel SPMD transform is **built, materializes a real
conv net, is numerically exact, and executes in real OpenMP** — all proven.  Its
*speedup* is workload-bound: near-linear on compute-bound kernels, ~1.4× on
memory-bound resnet50 inference (an inherent property of memory-bound CNNs, well
known in the literature).  The lever for a real CNN win is arithmetic intensity
(register-block/vectorize the convs first, then SPMD), not more parallelism.

## 11.12 Efficient compilation — no huge intermediates (2026-06-20)

The naive back-end wrote a textual file per stage: dr-opt `.mlir` (≈930 MB),
mlir-opt `.mlir` (≈930 MB), mlir-translate `.ll` (**≈3 GB** for gpt — weights as
decimal text, ~6× their binary size).  That overflowed `/tmp` (a 31 GB tmpfs)
and hit clang's source-location limit on the 3 GB `.ll`.

`scripts/validate-spmd-parallel.sh` now compiles in **one fully-piped pass with
binary IR**, zero intermediate files:

```
dr-opt … --emit-bytecode -o - \         # MLIR bytecode: weights stay BINARY
 | mlir-opt … --emit-bytecode -o - \     # (not re-serialized to text)
 | mlir-translate --mlir-to-llvmir -o - \ # textual .ll streams through the pipe
 | llvm-as -o - \                         # LLParser->bitcode (no clang text limit)
 | clang -O2 -march=native -c -x ir - -o k.o   # full middle-end + native codegen
```

- MLIR bytecode between dr-opt↔mlir-opt: binary weights, no 2× 930 MB text.
- The multi-GB `.ll` only ever streams through a pipe — never a file.
- `llvm-as` parses the textual `.ll` (LLParser has no source-location limit) →
  bitcode; clang compiles the **bitcode** (binary, no text limit) with the full
  `-O2` middle-end (vectorizer) + `-march=native` — `llc` alone skips the
  middle-end and is ~3× slower.
- Verified byte-equivalent + perf-equivalent to the old file-based clang path:
  resnet50 batch-1 1t=3.31 s, 16t=2.45 s (1.35×), `norm_rel_err 1.05e-6`; mnist
  `0.000e+00`, zero `.ll`/`.mlir` files left.  This also removes the disk /
  clang-frontend blockers that gpt's 3 GB `.ll` hit.

## 11. Open questions / honest limits

- **Shard-axis matching across layers** (which loop indexes the same buffer
  dimension) is heuristic for general IR; precise for ONNX-lowered kernels where
  activation tensors flow layer-to-layer. Start with the dominant-tensor
  heuristic; refine if it mis-picks.
- **Load imbalance**: static block partition assumes uniform per-shard work —
  fine for regular ML kernels, weak for ragged/triangular nests (work-stealing
  stays OUT; deterministic > balanced).
- **Halo cost** for deep stencil chains can exceed the barrier it replaces —
  the cost model gates per-edge, but the model needs the measured
  fork/join + barrier constants for the dev host first.
- **Inter-procedural** kernels (calls between layers): pure calls already
  consumed (M4); impure cross-layer calls need the deferred forwarding analysis.

## 11.12 Batch-1 speedup CLOSED — the cap was Amdahl, not bandwidth (2026-06-22)

§11.11 concluded resnet50 batch-1 SPMD was ~1.36×@16t and "memory-bound — an
inherent property of memory-bound CNNs."  **That diagnosis was wrong.**  The cap
was **Amdahl**: ~22 of 170 bands — including the heavy convs — fell to
`par.critical` (serial) because `par-spmd-perband`'s de-affine path could not
shard them, and the per-thread reductions were slow memref-accumulator
round-trips.  Both are now fixed; batch-1 scales **near-linearly**.

**Fix (3 edits to `ParBubbles.cpp`, the de-affine path):**
1. `cloneDeAffine` lowers `affine.for` *with iter_args* → `scf.for` carrying the
   iter_args (init operands, region iter args, `affine.yield`→`scf.yield`).  scf
   supports this natively; a promoted register-accumulator reduction now survives
   de-affining.  Sound: the shard axis is oracle-PARALLEL (owner-computes,
   disjoint output) and the accumulator is loop-local (iter_arg, not shared).
2. `deAffinable` accepts iter_arg reductions (was an explicit bail).
3. The de-affine path is no longer gated `shardIdx==0`.  Loops above the shard
   axis are guaranteed degenerate (extent≤step) by the shard-selection loop;
   `emitForall` maps their IVs to their lower bound (a sound trivial interchange
   past once-iterating loops), so a parallel axis BELOW a degenerate batch/extent-1
   loop — the resnet50 conv shape — shards.

**Pipeline:** add `dr-scalar-reduction-promote` after demote
(`func.func(dr-affine-loop-distribute,dr-scalar-reduction-demote,dr-scalar-reduction-promote)`).
Promote lifts the memref accumulator to an iter_arg register (no per-step DRAM
round-trip); the de-affine fix keeps that promoted band sharded.

**Result (resnet50-v2-7 batch-1, static `1x3x224x224`, krnl-free OpenMP back-end,
16-core Zen4, median of 7, `scripts/validate-spmd-parallel.sh`):**

| config | structure | t1 | t16 | scaling | vs plain-seq@16t | err |
|--------|-----------|----|----|---------|------------------|-----|
| plain-seq | (no SPMD) | 2.11s | 2.11s | — | 1.00× | 0 |
| demote→perband (old perband) | 148 fa / 22 crit | 3.34s | 2.44s | 1.36× | 0.86× | 1e-6 |
| demote→perband (**+fix**) | **168 fa / 2 crit** | 3.34s | 0.227s | **14.7×** | 9.3× | 1e-6 |
| demote+**promote**→perband (**+fix**) | **168 fa / 2 crit** | **2.21s** | **0.153s** | **14.5×** | **13.8×** | 1e-6 |

So the fix alone lifts the *existing* (demote-only) pipeline 1.36×→14.7× — proof
the cap was the 22 serial bands, not bandwidth.  Adding promote restores the fast
per-thread baseline (3.34s→2.21s ≈ plain-seq) on top, for **13.8× over the
equivalent sequential lowering**, numerically exact (err 1.05e-6 vs the seq
reference).  Lit: `test/Parallel/spmd-perband-reduction.mlir`; full suite 244/0
(16 pre-existing ONNX unresolved).

**Honest scope.** The 13.8× is the *parallelization* contribution measured against
the **same (scalar) codegen** sequential baseline — apples-to-apples, not a claim
vs onnx-mlir's vectorized `--O3` EmitObj.  Per-thread code is still non-vectorized:
`affine-register-block` would speed each thread further but currently turns the
stepped/unroll-jammed band back to `par.critical` (122/170 forall — the same
register-block × perband gap seen on PolyBench).  Closing THAT (shard the
register-blocked band) is the remaining frontier and would compound on top of the
14× parallel scaling.

## 11.13 openai-gpt (transformer) SPMD — NO-GO, serial-bound (2026-06-23)

> **SUPERSEDED by §11.14 (same day).** The "serial-bound" diagnosis below was
> WRONG — the serialization was an `ParAliasOracle` bug (an in-loop `memref.alloca`
> scalar accumulator was not privatized, so the GEMM row/col axes were marked
> SEQUENTIAL(conservative) and the heavy QKV/FC GEMMs fell to `par.critical`).
> With the oracle fixed, openai-gpt SPMD scales **18.0× @16t** (err 0). Kept for
> the record; read §11.14 for the real result.

Applied the resnet50-winning pipeline (demote+promote → par-spmd-perband →
par→omp, with the §11.12 de-affine fix) to openai-gpt
(`openaigpt_Opset18.onnx`, static 1×128, two inputs: input_ids i64 +
attention_mask f32 → hidden f32[1,128,768]). Repro:
`scripts/validate-spmd-openaigpt.sh`.

**Structurally it parallelizes** — 434/519 bands forall (84%), and promote
composes (V0==V1, the de-affine fix carries over from convs to the transformer's
GEMM/projection reductions). **But it does NOT scale:**

| model | bands forall | scaling @16t | err |
|-------|--------------|--------------|-----|
| resnet50 (convnet)    | 168/170 (99%) | **14.5×** | 1e-6 |
| openaigpt (transformer) | 434/519 (84%) | **1.01×** | 0 |

plain-seq 20.5s; spmd 1t 21.2s / 16t 21.1s — flat.

**Diagnosis (measured, not guessed).** At 16 threads the OpenMP team spins
~600–700% CPU (not 1600%) with ZERO wall-time improvement — ~147 CPU-s to do
21 CPU-s of work. `OMP_WAIT_POLICY=passive` leaves wall-time at 21.0s. So the
700% CPU is barrier-spin waste; the forall bands hold ~none of the runtime, and
wall-clock = the **serial critical path**:
- **85 critical bands** = the genuine reductions (softmax max/sum over keys,
  LayerNorm mean/var over hidden) that don't shard owner-computes;
- **518 barriers** between 519 small bands — per-band parallel work is too small
  to amortize the sync;
- scalar transcendentals (Gelu `powf`+`tanh`, softmax `exp`) dominate compute.

**Why convnet ≠ transformer for SPMD.** resnet50's convs are a few large dense
parallel chunks (each forall is millions of MACs → barriers amortize). The
transformer's runtime lives in serial reductions + scalar transcendentals spread
over many tiny bands. Whole-function per-band SPMD exposes parallel *bands* but
not parallel *work*.

**Verdict.** SPMD is a CONVNET lever, not a transformer lever. This reinforces
the prior transformer findings (`ONNX_O3_GAP_OPENAIGPT.md`,
`TRANSFORMER_ELTWISE_SPEC.md`): openai-gpt's lever is **per-thread codegen** —
the naive 20.5s seq drops ~70× to ~0.29s under the demote+register-block+promote
codegen path (T1–T5c), where the remaining headroom is the scalar transcendentals
(Gelu `powf(x,3)` never strength-reduced ~20–45% of runtime) and attention
fusion, NOT parallelism. Barrier elision in perband would cut the spin waste but
not the serial critical path, so it does not rescue scaling here.

## 11.14 CORRECTION — openai-gpt SPMD scales 18× (the §11.13 cap was an oracle bug) (2026-06-23)

§11.13 concluded the transformer was "serial-bound, SPMD is a convnet-only lever."
That was wrong, and the giveaway was exactly the right question: a transformer is
full of independent work (every GEMM row, every head, every token) — so why would
it be serial?

**Root cause.** `ParAliasOracle::axisConflict` saw the per-output `memref.alloca`
scalar accumulator that demote/promote leaves on a GEMM+bias band, and (a) its
allocation effect set `sawOpaque` → `Unknown`, and (b) its scalar load/store
looked like a same-address carried dependence. So the GEMM's **row (m=128) and
col (n=2304/3072) axes were marked `SEQUENTIAL(conservative)`** and the heavy
QKV/FC GEMMs — the bulk of the FLOPs — fell to `par.critical` (serial). At 16
threads the team then spun ~700% CPU at the 518 barriers with no progress, which
looked like an inherent serial bound but was Amdahl on mis-classified GEMMs.

**Fix** (`lib/Analysis/ParAliasOracle.cpp`, `axisConflict`): an alloca created
INSIDE the classified loop is loop-private — a fresh allocation per iteration that
never escapes the band (only its loaded value is stored to the owner-computes
output). Privatize it: exclude its accesses from the dependence test and don't
treat its allocation as opaque. Sound — distinct per-iteration memory carries no
cross-iteration dependence, and replicating thread-private stack scratch is race-
free. (Also `deAffinable` accepts `memref.alloca` so the de-affine emit clones it
per shard.)

**Result.** openai-gpt critical bands **85 → 37** (forall 434 → 482; the 37 left
are the genuine softmax max/sum + LayerNorm mean/var reductions). E2E
(`scripts/validate-spmd-openaigpt.sh`, batch-1, 16-core Zen4, median of 7):

| | t1 | t16 | self-scaling | vs plain-seq@16t | err |
|--|----|----|--------------|------------------|-----|
| openai-gpt SPMD (was §11.13) | — | — | — | 1.01× | 0 |
| openai-gpt SPMD (oracle fix) | 20.16s | **1.09s** | **18.5×** | **18.0×** | 0.00e+00 |

Numerically EXACT (byte-identical to the sequential reference). resnet50 also
improved 168/2 → 169/1 forall (no regression, e2e re-validated). Lit:
`test/Parallel/spmd-perband-alloca-scratch.mlir`; full suite 246/0 (16 pre-existing
ONNX unresolved).

**Corrected verdict.** Whole-function SPMD IS a transformer lever after all —
openai-gpt scales 18× (the GEMMs + pointwise are owner-computes parallel; only the
true reductions stay serial). The earlier "convnet-only" claim was a measurement
artifact of the oracle conservatism, not a property of transformers. This is
ORTHOGONAL to the per-thread codegen lever (register-block + transcendentals,
§ONNX_O3_GAP): SPMD parallelizes across threads, codegen speeds each thread —
they compose (each parallel shard would run the vectorized per-thread kernel).

## 11.15 SPMD vs onnx-mlir native `--parallel` (batch-1) — split decision (2026-06-23)

Head-to-head against onnx-mlir's own `--parallel` (its native EmitObj backend +
OpenMP), batch-1, 16-core Zen4, 16 threads.  Scripts:
`scripts/spmd-vs-native-parallel-{resnet50,openaigpt}.sh` (native EmitObj +
OMTensor harness; the `--parallel` object needs libomp `__kmpc_*`, mounted from
the host since neither onnx-mlir image ships it).

| model | onnx-mlir `--O3` (vec, seq) | onnx-mlir `--O3 --parallel` @16t | our SPMD @16t (scalar/thread) | winner |
|-------|---------------------------|----------------------------------|-------------------------------|--------|
| resnet50  | 1.29s | 1.44s (**0.99×**, did NOT parallelize) | **0.152s** (14×) | **SPMD, 8.5×** |
| openai-gpt | 0.58s | **0.120s** (6.52×, vec+par) | 1.09s (18×) | **onnx-mlir, 9.1×** |

**onnx-mlir `--parallel` is inconsistent at batch-1**: it scales the transformer
(6.5×) but completely fails resnet50 conv (0.99× — flat). Our SPMD scales BOTH
(14× / 18×), so on *coverage* SPMD wins. (`--EmitMLIR` shows 0 `krnl.parallel`
for both; misleading — the parallelization is applied in the `--EmitObj`
lowering, so the binary is authoritative, not the krnl dump.)

**The absolute winner flips on vectorization headroom, because onnx-mlir composes
VEC + PAR and our SPMD is PAR-only (scalar per thread):**
- resnet50: conv vec headroom is small (1.29s vec vs 2.11s our scalar = 1.6×), so
  our 14× parallel beats their (failed-parallel) vectorized seq → SPMD wins 8.5×.
- openai-gpt: transformer vec headroom is huge (0.58s vec vs 20.5s our scalar =
  35×), so even our 18× parallel loses to their vec+par → onnx-mlir wins 9.1×.

**Conclusion.** The decisive lever is not parallelism — it is composing
vectorization WITH parallelism. onnx-mlir does both; our SPMD does only the
parallel half (scalar per-thread, because `affine-register-block` currently turns
the stepped/unroll-jammed band back to `par.critical` — the register-block ×
perband gap). Closing that — register-block / transcendentals INSIDE each shard —
is the one change that would beat onnx-mlir on BOTH (≈14× parallel × ≈30×
per-thread vec headroom on the transformer). That is the next work item (§11.16).

## 11.16 SPMD × vectorization compose — mechanism LANDED, gated by register-block coverage (2026-06-23)

The §11.15 conclusion: to beat onnx-mlir we must compose vectorization WITH
parallelism. `affine-register-block` (the per-thread vectorizer) emits a stepped,
unroll-jammed band with `affine.vector_load/store` + `vector<Nxf32>` iter_args;
`par-spmd-perband` used to bail that to `par.critical` (the register-block ×
perband gap, §11.12). 

**Fix (ParBubbles.cpp):** `deAffinable` accepts `affine.vector_load/store`, and
`cloneBodyOp` lowers them to `vector.load/store` (same affine-map expansion as the
scalar case). So a register-blocked band shards into a `par.forall` whose body is
the vectorized kernel — each SPMD shard runs the vectorized micro-kernel.

**Mechanism proven** (constant-bound gemm N=1024, mlir-runner --O3,
checksum-MATCH): seq 1t; rb 1t; par-spmd 16t; **rb+par-spmd 16t — both fire
(1 wsloop, vector ops preserved), numerically exact.** The compose is
near-multiplicative where register-block fully fires. (In-house now; no upstream
`affine-parallelize` fallback as PolyBench needed.) Lit:
`test/Parallel/spmd-perband-register-blocked.mlir`; Parallel suite 21/21.

**But on the real models it is currently NEUTRAL — gated by register-block
COVERAGE, not the compose.** openai-gpt rb+par-spmd: 482 forall + 576 vector ops
(both fire), e2e 16t = **1.085s ≈ par-spmd-only 1.09s** (err 2.95e-6). The
register-block vectorizes only minor bands; the **dominant QKV/FC GEMMs stay
scalar** because they are in onnx-mlir's `iter_args`/alloca-accumulator/GEMV form
that plain `affine-register-block{mr,nr}` does not catch (needs the transformer
campaign's `canonicalizeAllocaGemm`; convs need `vectorizeConvBand`). So the
compose multiplies a ~1× per-thread win → no e2e change, still 1.085s vs
onnx-mlir's 0.120s.

**Status.** Compose mechanism: DONE, correct, tested, removes the §11.12
limitation. Beating onnx-mlir on the transformer now reduces to a *register-block
coverage* problem (vectorize onnx-mlir's GEMM/conv forms), which the compose then
multiplies by the 18× SPMD factor. That coverage work is the next frontier.
