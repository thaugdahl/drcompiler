# PARALLEL_BUBBLE_SPEC.md — Bubble-Widening Region Formation

Status: design spec (v1, 2026-06-18). Companion figures:
`docs/parallel-codegen/bubble-widening.typ`.

This spec defines the **region-formation front-end** for drcompiler's parallel
codegen track: identify *maximal* regions of code that can execute concurrently
across hardware threads by **iterative widening** of per-loop "bubbles", resolve
conflicts (peel / privatize / redistribute), and materialize each maximal region
into a new runtime-agnostic parallel dialect (`par`).

It is the **complement** to `PARALLEL_CODEGEN_SPEC.md` (the 2026-06-13 sharding
spec): that spec is the *back-end* — `dr-shard`/`libdrpar`, the pinned-thread
runtime, `decideShard()`, and the topology/placement model. Bubble-widening decides
*what* is one parallel region and *how its axes map to workers*; the existing
back-end decides *how many cores, what grain, which cache domain* and emits the
pinned runtime calls. The `par` dialect (§6) is the contract between them.

---

## 0. Scope & locked decisions

| # | Decision | Consequence |
|---|----------|-------------|
| 1 | **Affine-first, SCF-supported**, unified by an **intra-module alias analysis** | Polyhedral-exact legality where accesses are affine (`checkMemrefAccessDependence`); conservative alias elsewhere. One oracle (§2). |
| 2 | **Static compile-time legality only** | No runtime-versioned / speculative loops in v1. Cannot prove independence ⇒ axis stays sequential. Speculation is future work (§13). |
| 3 | **Emit a new runtime-agnostic parallel dialect (`par`)** | Materialization stops at `par`; lowering `par → libdrpar / omp / scf.parallel` is a **separate phase (Phase B)** (§6). |
| 4 | **Parallel-loops-only core; reductions are stretch** | v1 distributes loops with no carried dependence on the distributed axis. GEMM/conv reduction-split = M5/stretch (§8), and reuses the back-end's existing reduction sharding. |
| 5 | **Layered atop the dr-shard back-end** | The `par → libdrpar` lowering reuses `decideShard()`, the ctx-struct outlining, the pinned pool, and the `Topology` model from `PARALLEL_CODEGEN_SPEC.md`. Bubble-widening replaces the *single-outermost-loop matcher* of `dr-shard` with maximal-region formation; everything downstream of `par.region` is the existing back-end. |

Out of scope (v1): NUMA placement, GPU, dynamic/inspector-executor, task (non-loop)
parallelism, distributed memory. The NUMA / heterogeneous-placement / false-sharing
concerns are **owned by the back-end** (`PARALLEL_CODEGEN_SPEC.md` §5) and inherited
by the `par → libdrpar` lowering — this spec does not re-specify them.

---

## 0.5 Relationship to `PARALLEL_CODEGEN_SPEC.md` (the layering)

```
            ┌───────────────────────── THIS SPEC (front-end) ─────────────────────────┐
  affine/   │  dr-par-bubbles:  seed bubbles → widen to fixed point → peel/redistribute │
  scf IR ──▶│                       → materialize MAXIMAL regions                       │──▶ par.region
            └───────────────────────────────────────────────────────────────────────┘      par.forall (mapping)
                                                                                             par.{barrier,redistribute,critical,reduce}
            ┌───────────────────── PARALLEL_CODEGEN_SPEC.md (back-end) ──────────────────┐
  par IR ──▶│  par → libdrpar lowering (Phase B):  decideShard() cores/grain/domain      │──▶ __drpar_for(...)
            │  reuse ctx-struct outlining, pinned pool, Topology placement (§4–§6 there)  │     (pinned, sharded)
            └────────────────────────────────────────────────────────────────────────┘
            (alt sinks: par → omp.parallel/wsloop  [oracle/portability];  par → scf.parallel [testing])
```

What changes vs the standalone `dr-shard`:
- `dr-shard` matched and sharded **one** outermost `isLoopParallel` loop per hot
  nest. Bubble-widening instead **grows** a maximal region across multiple loops,
  siblings, peeled slabs, and (M4) calls — then a `par.region` is the unit handed
  to the back-end. A maximal region may carry **several** distributed axes
  (`par.forall` with >1 IV), so `decideShard()` is invoked per region over its full
  distributed iteration space rather than per single loop.
- The transform *mechanics* `dr-shard` defined — ctx-struct outlining, the pinned
  pool ABI (`__drpar_for`/`__drpar_alloc_local`/`__drpar_reduce_f64`), static block
  partition, `decideShard()` (`cores`/`grain`/`domain`/`padBoundary`), the
  `Topology` model, false-sharing padding, NUMA first-touch — are **unchanged** and
  reused inside the `par → libdrpar` lowering.
- New constructs bubble-widening contributes that `dr-shard` had no IR for:
  `par.redistribute` (transitory region between non-conformant parallel regions,
  §4.2) and `par.critical` (peeled serial slab, §4.1). The back-end lowers these as
  pool barriers + a reshuffle / single-worker section.

Net: **one pipeline slot** (after `register-block`/`fission`, where `dr-shard` sat),
now `dr-par-bubbles` + the `par` lowerings. Default OFF / byte-identical (§7, §10).

---

## 1. Terminology

- **Bubble** — an in-memory analysis object (C++, *not* IR) owning a contiguous
  single-entry/single-exit region of ops plus the loop axes it has proven parallel.
  Seeded one per loop; widens by consuming neighbours; coalesces toward maximal
  parallel regions.
- **Frontier** — ops/regions adjacent to a bubble it may consume next: its enclosing
  loop (*climb*), preceding/following siblings (*engulf*), adjacent sibling bubbles
  (*fuse*).
- **Parallel axis (`parDim`)** — a loop IV along which the body carries no dependence
  (or only a recognized reduction kept sequential). Becomes a distributed dimension
  of the final `par.forall`.
- **Sequential axis (`seqDim`)** — an axis that must run in order within each worker
  (carried dependence, or reduction in v1).
- **Conflict** — a dependence whose direction has a non-zero component along a
  candidate `parDim` (would be violated by distributing that axis).
- **Transitory region / redistribution** — a `par.redistribute` boundary between two
  parallel regions whose worker→data mappings are incompatible (different trip
  counts, transposed layout): keep both parallel, reshuffle the mapping between them.
- **Peel** — split a loop's iteration domain (affine set difference) into a
  conflict-free interior that stays parallel and a small conflicting slab emitted
  sequentially (`par.critical`) — §4.1.

---

## 2. IR substrate and the alias oracle (`ParAliasOracle`) — net-new

There is **no** `AliasAnalysis` user anywhere in the tree today (confirmed); only
affine dependence is used (`lib/Transforms/DrAffineLoopDistribute.cpp:218`,
`lib/Transforms/AffineLoopFusionCostModel/LoopFusion.cpp:577`). So `ParAliasOracle`
is net-new infra in `lib/Analysis/` (links `MLIRAffineAnalysis`, `MLIRAffineUtils`,
reuses `DRCompAnalysis`). It is the single legality predicate the widening loop calls:

```
ConflictKind mayConflictUnderParallel(const Access &a, const Access &b, AxisRef axis);
//   ⇒ None | LoopIndependent | Carried | Unknown
//  Access = { Operation *op; Value memref; AffineRel rel | nullptr; }
```

Tiered; first tier to decide wins:

- **Tier 0 — provenance disjointness.** Distinct allocation roots never alias. Reuse
  `dr::AnalysisContext::allocRootFor` / `AllocationRoots`
  (`include/drcompiler/Transforms/DataRecomputation/AnalysisState.h:59,130`).
  Function-arg memrefs: distinct iff a `restrict`/no-alias attr is present *or*
  call-graph forwarding (§5) proves distinct roots; else may-alias → fall through.
- **Tier 1 — affine exact.** Same base memref, both affine: `mlir::affine::MemRefAccess`
  + `checkMemrefAccessDependence` collecting `SmallVector<DependenceComponent>` per
  loop depth (the exact pattern at `LoopFusion.cpp:573-581`). The component along
  `axis`'s depth gives `None` / `LoopIndependent` (zero-distance, privatizable) /
  `Carried`. `FlatAffineValueConstraints` (`LoopFusion.cpp:797`) backs the §4 peel.
- **Tier 2 — conservative.** Non-affine subscripts, `scf.for` dynamic bounds, or
  unknown-provenance memref ⇒ `Unknown` unless Tier 0 separated them. `Unknown` is
  treated as a conflict (sound).
- **Cross-procedure.** Map callee accesses to caller memrefs via `EnrichedCallGraph`
  (`AnalysisState.h:101`) + `InterproceduralOrigin` (`:112`) before Tiers 0–2 (§5).

### 2.1 The `Bubble` structure

```cpp
struct Bubble {
  unsigned id;
  Operation *seed;                       // originating affine.for / scf.for
  llvm::SetVector<Operation *> body;     // ops currently claimed (SESE region)
  SmallVector<Value> parDims;            // IVs proven parallel  → par.forall axes
  SmallVector<Value> seqDims;            // carried/reduction IVs → inner sequential loops
  AccessSummary reads, writes;           // {memref → AffineRel ∪ PointSet}
  llvm::DenseSet<Operation *> barriers;  // sync points inside
  SmallVector<PrivatizationPlan> privatized;
  SmallVector<PeelPlan> peels;           // §4.1
  SmallVector<RedistEdge> redist;        // §4.2
  enum class State { Active, Frozen } state = State::Active;
};
```

`PointSet` (`include/drcompiler/Transforms/DataRecomputationIndexing.h:24`,
`overlaps`/`+=`/`-=`) is the constant-index fast path for footprint overlap; the
general case uses the affine relation.

---

## 3. The widening algorithm

### Phase 0 — Seed

For every `affine.for`/`scf.for`: build a `Bubble`, gather body reads/writes, and
classify each loop depth `d` by running `mayConflictUnderParallel` over all
same-memref access pairs at depth `d`:

- all `None`/`LoopIndependent` → `parDim`;
- a `Carried` that is a recognized associative reduction → `seqDim` tagged
  `reduction` (M5+; v1 keeps it sequential);
- any other `Carried`/`Unknown` → `seqDim`.

### Phase 1 — Iterative widening to fixed point

```
worklist ← all Active bubbles, innermost-first        // inner bubbles stabilize first
repeat
  changed ← false
  for B in worklist where B.state == Active:
    for move in frontierMoves(B):                     // Climb | EngulfPred | EngulfSucc | Fuse
      verdict ← evaluate(B, move)                     // ParAliasOracle + cost gate (§7)
      switch verdict.kind:
        Clean:        apply(B, move);                 changed ← true
        Resolvable:   if costGate(verdict.fix):                      // privatize | peel | redistribute
                        apply(B, move, verdict.fix);  changed ← true
                      else freezeEdge(B, move)
        Hard:         freezeEdge(B, move)             // barrier / unresolvable dep / opaque call
                      if move is Fuse: recordRedistBoundary(B, move.other)
  if !changed: break
until fixed point
```

**Frontier moves**

1. **Climb** — let `L` enclose `B`'s outermost claimed op. Add `L` as a `parDim` iff
   it carries no conflict over `B`'s reads/writes at `L`'s depth. A `Carried`
   reduction → `L` becomes a `seqDim`.
2. **EngulfPred / EngulfSucc** — a sibling op/region immediately before/after `B`.
   Legal iff every dependence between it and `B` is `None`/`LoopIndependent` under
   `B`'s `parDims`.
3. **Fuse** — adjacent sibling `Bubble B'`:
   - **Conformant** (compatible trip counts / worker mapping) and dependence-clean →
     merge (union bodies, intersect `parDims`).
   - **Non-conformant** but both parallel → do *not* merge; record a `RedistEdge`
     (`par.redistribute`, §4.2).

**Conflict outcomes**

- **Clean** — apply directly.
- **Resolvable** — fixable by *privatization* (per-worker copy of a false-dep
  scalar/temp), *reduction recognition* (M5+), or *peel* (§4.1). Each fix is gated by
  the cost model (§7): accept only if the modelled penalty (privatization memory,
  serial slab work, redistribution bytes) keeps the region net-positive.
- **Hard** — true barrier-like sync, unresolvable carried dep, or opaque/aliased
  call. Freeze the edge; two bubbles meeting at a hard edge stay separate
  `par.region`s with a sync boundary.

**Fixed point** — a full sweep with `changed == false`. Each bubble is now maximal
under the static oracle.

### Phase 2 — Materialize

Each bubble whose cost gate is positive → a `par.region` (§6). Bubbles that never
grew past a single sequential loop, or that fail the gate, are left as ordinary
affine/scf loops — no `par` ops emitted (default machine stays byte-identical, §7).

---

## 4. Fine-grained control: peeling & redistribution

### 4.1 Peeling a partial conflict

A dependence often covers only a sub-polyhedron. Canonical offset-`K` case:

```
affine.for %i = 0 to N { store X[%i] }        // bubble A
affine.for %i = 0 to N { use   X[%i + K] }     // bubble B  — conflict only in a K-slab
```

Resolution: (1) build the conflicting sub-domain via `FlatAffineValueConstraints`
intersection (constant-index fast path: `PointSet::overlaps`/`-=`); (2) split —
**interior** stays in the `par.forall`, **boundary slab** peels to a sequential
`par.critical` prologue/epilogue; (3) gate: peel only when `interior ≫ slab`. This is
the spec's "resolve affine map overlaps — peel critical regions" requirement.

### 4.2 Redistribution (transitory regions)

Two regions both parallel but with disagreeing worker↔data mappings (row-parallel
producer → column-parallel consumer, or mismatched trips): emit `par.redistribute`
between the `par.region`s — an explicit reshuffle of the thread→data assignment,
rather than forcing a serializing fuse. Its modelled cost (`redistBytes /
redistBytesPerCycle`, §7) is what the gate weighs against keeping both parallel. The
`par → libdrpar` lowering realizes it as a pool barrier + a `__drpar_alloc_local`
re-placement (reuses the back-end's first-touch path,
`PARALLEL_CODEGEN_SPEC.md` §5.2).

---

## 5. Interprocedural call consumption

A `func.call` on a bubble's frontier triggers **call-site + forwarding analysis**,
reusing existing interprocedural machinery (no new escape analysis):

1. Map actuals → formals (`EnrichedCallEdge`, `AnalysisState.h:94`).
2. Translate the callee's per-arg effects (`CalleeArgEffect`, `:105`) into *caller*
   memrefs via `InterproceduralOrigin` (`:112`) / `InterproceduralOriginMap` (`:119`);
   module-global writes from `ModuleGlobalWrites` (`:126`).
3. Decide:
   - **Pure / per-iteration-disjoint** → consumable: kept inside the region as
     `par.call` (inlined only if small + cost-positive).
   - **Synchronizing / aliased / unknown** → hard boundary; freeze here.

The DataRecomputation provenance classes (SINGLE/MULTI/LEAKED/KILLED, derived from
`StoreSet` shape — `Passes.td:15-20`) are the backstop: a callee argument whose
caller provenance is `LEAKED` forces a hard boundary.

---

## 6. The `par` dialect (the contract; full design = Phase B)

Runtime-agnostic: names *what* is parallel and *how data maps to workers*, never a
runtime. Mirrors the ODS convention in `include/drcompiler/Dialect/Topology/IR/` but
**fixes the wiring that orphaned scaffold lacks** (§10).

| Op | Role |
|----|------|
| `par.region` | Maximal parallel region; isolated-from-above, explicit captures; holds the worker `mapping` attr. The unit handed to `decideShard()`. |
| `par.forall` | Distributed iteration space over abstract workers (≥1 `parDim` IV). `mapping` attr (block/cyclic/collapsed); lowering picks the schedule. |
| `par.barrier` | Synchronization point. |
| `par.redistribute` | Transitory region: reshuffle thread↔data mapping (§4.2). |
| `par.critical` | Peeled/ordered single-worker sub-region (§4.1). |
| `par.reduce` | Cross-worker associative reduction (M5/stretch, §8). |
| `par.yield` | Terminator. |

**Lowerings (Phase B, own spec):**
- `par → libdrpar` **(primary)** — reuses `decideShard()` (cores/grain/domain/
  padBoundary), the ctx-struct outlining, the pinned pool, `Topology` placement, and
  reduction sharding from `PARALLEL_CODEGEN_SPEC.md`. A multi-axis `par.forall`
  presents its collapsed/blocked iteration space to `decideShard()`.
- `par → omp.parallel`/`omp.wsloop` — correctness oracle + portability fallback
  (the back-end spec already keeps OpenMP as the oracle, §7 there).
- `par → scf.parallel` — testing sink (M2).

Skeleton (placeholder name `par`, knob §13):

```tablegen
def Par_Dialect : Dialect {
  let name = "par";
  let cppNamespace = "::mlir::par";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}
class Par_Op<string mnemonic, list<Trait> traits = []> : Op<Par_Dialect, mnemonic, traits>;
```

---

## 7. Cost model: the per-region gate (reuse `decideShard()`)

The gate is **not** new infra — it is `decideShard()` from `PARALLEL_CODEGEN_SPEC.md`
§6, invoked per maximal region over its distributed iteration space:

```
struct ShardPlan { unsigned cores; int64_t grain; int domain; bool padBoundary; };
ShardPlan decideShard(regionTrip, perIterComputeCycles, perIterBytes, const MachineModel &mm);
```

A region is materialized to `par` only when `decideShard().cores > 1`. Per-iteration
work sums `CpuCostModel::opCost` (`CpuCostModel.h:90`); `MachineModel`
(`MachineModel.h:32`) carries the `ThreadModel` + roofline + `Topology`.

Two gaps this front-end must close on top of the back-end model:

1. **Spawn/barrier term.** `MachineModel::ThreadModel` (`MachineModel.h:121`) models
   parallelism only as bandwidth division — *no* spawn/barrier latency (confirmed).
   `dr-shard` amortized fork/join implicitly via `minGrain` (a §6 calibration
   constant). For multi-region widening (which creates *many* regions + barriers +
   redistributes) make it explicit; add to `ThreadModel`, defaulting 0 so the
   default machine stays byte-identical (mirrors the `hasExplicitThreadModel` guard):

   ```cpp
   unsigned spawnCycles   = 0;   // fork+join per par.region (0 ⇒ not modeled)
   unsigned barrierCycles = 0;   // one team barrier
   double   redistBytesPerCycle = 0;   // par.redistribute bandwidth (§4.2)
   ```

2. **Redistribution cost.** `par.redistribute` has no analogue in the single-loop
   `decideShard()`; price it as `redistBytes / redistBytesPerCycle` and fold into the
   region's gate (a redistribute is profitable only if both adjacent regions stay
   parallel and the reshuffle is cheaper than serializing one of them).

> Conservative default: with `spawnCycles == 0` and no explicit thread model,
> `decideShard()` returns `cores == 1`, so **no `par.region` is emitted**. The pass is
> inert-by-default and byte-identical, consistent with every drcompiler pass and with
> `dr-shard`'s default-OFF discipline.

---

## 8. Reductions (M5 / stretch)

v1 keeps reduction axes in `seqDim` (each worker reduces its tile sequentially). The
GEMM/conv case (`C[i,j] += A[i,k]*B[k,j]`) distributes parallel `i,j` and leaves `k`
sequential per worker — the shape `affine-register-block` already produces, and the
`sOut` loop `dr-shard` already shards. Stretch:

- Recognize associative reductions (reuse detection from `dr-scalar-reduction-demote/
  -promote`, `Passes.td:208-269`).
- Emit `par.reduce` for the no-outer-parallel-axis case (GEMV, dot, norms); the
  `par → libdrpar` lowering realizes it with the back-end's **existing** per-thread
  padded partial accumulators + `__drpar_reduce_f64` tree-combine
  (`PARALLEL_CODEGEN_SPEC.md` §3.3), under the same `fastmath<reassoc>` and
  `--par-deterministic` controls.

---

## 9. Soundness

A bubble's distributed execution along its `parDims` is legal iff, for every memref
dependence with a non-zero component on a `parDim` axis, one holds: (a) it is absent
(Tier 0/1 `None`); (b) it is a recognized associative reduction kept in `seqDims`
(M5+); (c) it is privatized; or (d) it is peeled to a `par.critical`/sequential slab
or a `par.redistribute`. Anything else — every `Unknown`, every `LEAKED` callee arg
— forces the axis sequential. No speculation, no runtime checks: a strict refinement
of the original dependence order. (Matches `PARALLEL_CODEGEN_SPEC.md` §7: only
provably-parallel work is distributed; a miss is a perf miss, never a correctness bug.)

---

## 10. Pass pipeline, CLI, dialect & build wiring

### 10.1 Pass (`include/drcompiler/Transforms/Passes.td`, existing convention)

`dr-par-bubbles` (`mlir::ModuleOp`) — the widening pass, in the slot `dr-shard`
occupied (after `register-block`/`fission`). Options (sketch):

- `par-test-diagnostics : bool` — per-bubble remarks (testing, §11).
- `par-materialize : bool` (default `false`) — emit `par` ops; else analysis-only (inert).
- `par-enable-peel : bool`, `par-enable-interproc : bool` — per-milestone gates.
- `par-max-redistribute : unsigned` — cap transitory regions per merge.
- `cpu-cost-model-file : std::string` — same JSON dispatch as the other passes
  (sources the `ThreadModel`/`Topology` from `PARALLEL_CODEGEN_SPEC.md` §4).
- `dependentDialects = [affine, scf, memref, func, par]`;
  `constructor = "mlir::createParBubblesPass()"`.

Auto-registered via `-gen-pass-decls -name DRCompPasses`
(`include/drcompiler/Transforms/CMakeLists.txt:2`) → `registerDRCompPassesPasses()`,
already called at `tools/dr-opt/dr-opt.cpp:20`.

### 10.2 Dialect wiring (fix what the Topology scaffold got wrong)

1. `include/drcompiler/Dialect/Par/IR/{Par.td, ParOps.td, CMakeLists.txt}` mirroring
   `Dialect/Topology/IR/` — but use `add_mlir_dialect`, and write decls/defs to
   **distinct** `.inc` files (Topology wrote both to `Topology.h.inc`).
2. `lib/Dialect/Par/IR/{ParDialect.cpp, ParOps.cpp}` — a real `add_mlir_library`
   (Topology has *no* `lib/Dialect` at all).
3. **Add the missing `add_subdirectory` chain to the top-level `CMakeLists.txt`** (it
   adds only `include/drcompiler/Transforms`, `lib/Analysis`, `lib/Transforms`,
   `tools` — not the `Dialect` trees).
4. Register in `tools/dr-opt/dr-opt.cpp` (`registry.insert<par::ParDialect>()`).
5. Avoid the scaffold's bugs: namespace typo (`toplogy`), duplicate-`.inc` filename,
   `add_mlir_doc` "TopoplogyOps" typo.

---

## 11. Testing strategy

Mirror the `dr-test-diagnostics` lit pattern (`-verify-diagnostics` + `expected-remark`):

```mlir
// RUN: dr-opt %s --pass-pipeline='builtin.module(dr-par-bubbles{par-test-diagnostics})' -verify-diagnostics
func.func @two_loops(%X: memref<?xf32>) {
  // expected-remark @below {{bubble 0: parallel axes {i}; seq {}}}
  affine.for %i = 0 to 1024 { ... }
  // expected-remark @below {{frontier: fuse bubble 0 -> REDISTRIBUTE (offset K conflict; peel slab=K)}}
  affine.for %i = 0 to 1024 { ... }
}
```

Matrix: (1) embarrassingly-parallel single loop; (2) climb through a perfect nest;
(3) offset-`K` peel; (4) non-conformant fuse → redistribute; (5) carried dep →
sequential; (6) pure-call consume vs aliased-call hard boundary; (7) cost-gate inert
by default (no `par` ops, no thread model). Materialization → `FileCheck` on `par`
ops. Lowering parity → diff `par → omp` against `par → libdrpar` (the back-end's
OpenMP oracle, §7 there). Add to `check-drcompiler`.

---

## 12. Milestones

| M | Deliverable | Gate |
|---|-------------|------|
| **M0** | `ParAliasOracle` (Tiers 0–2) + seeding + per-axis classification. `dr-par-bubbles{par-test-diagnostics}` remarks. No IR mutation. | lit remarks match (1),(2),(5). |
| **M1** | Widening to fixed point: Climb + Engulf + conformant Fuse; Clean/Hard only. Cost gate via `decideShard()`. | Fixed point reached; remarks show maximal regions. |
| **M2** | `par` dialect (Phase B) + materialization + `par → scf.parallel` (test sink). | FileCheck on `par.region`/`par.forall`; default byte-identical. |
| **M3** | Affine-overlap peeling + `par.redistribute`. | Tests (3),(4). |
| **M4** | Interprocedural call consumption. | Test (6). |
| **M5** (stretch) | Reduction recognition + `par.reduce`; GEMM/conv reduction-split. | GEMM distributes i,j; k sequential. |
| **M6** | `par → libdrpar` lowering: reuse `decideShard()` + pinned pool + topology. `par → omp` oracle. | End-to-end scaling vs sequential and vs `dr-shard` single-loop; OpenMP-oracle parity. |

---

## 13. Open knobs (chosen defaults — flag, not blockers)

- **Dialect name** — `par` (alt `drpar` if upstream-clash risk). C++ ns `::mlir::par`.
- **Worklist order** — innermost-first.
- **Consumed-call handling** — keep as `par.call`; inline only if small + cost-positive.
- **Nested parallelism** — allow nested `par.region`; collapse-vs-nest deferred to the
  `par → libdrpar` lowering (the back-end's §3.4 collapse already handles batch-1).
- **Privatization scope (v1)** — scalars + small fixed-size temps; array expansion deferred.
- **Cost-gate threshold** — `decideShard()` + the new `spawnCycles`/`barrierCycles`/
  `redistBytesPerCycle` terms; needs measured fork/join + barrier constants for the
  dev host (the back-end's §6 calibration) before M2 gating is trusted.

### Resolved
- **Relationship to `PARALLEL_CODEGEN_SPEC.md`** — *layered*: bubble-widening is the
  region-formation front-end; `dr-shard`/`libdrpar`/`decideShard`/topology is the
  back-end consuming `par.region` (decision #5, §0.5).

### Deferred (future work)
- Runtime-versioned / inspector-executor speculation (decision #2).
- NUMA / heterogeneous placement / false-sharing — owned by the back-end spec.
- GPU lowering, distributed memory, task parallelism.
```