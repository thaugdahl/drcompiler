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
```