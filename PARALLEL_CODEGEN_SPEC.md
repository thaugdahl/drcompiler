# PARALLEL_CODEGEN_SPEC — loop sharding + pinned execution for drcompiler

Status: SPEC (2026-06-13). Follow-up to `CROSSCUTTING.md` (the cross-thread cost
model) — that work gave the compiler a model of parallel *deployment*
(`ThreadModel`: `activeThreads`, `exclusive`/interspersed, per-level bandwidth,
SMT) but **no way to emit parallel code**. Every cross-thread decision is
currently advisory: the model picks the best *serial* code for a parallel target.
This spec adds the backend that makes those decisions real — it **shards
data-parallel loops across pinned threads** — and lays out the concerns that turn
"runs on N cores" into "specialized, predictable performance."

Methodology unchanged: spike-first, honest verdicts, one local commit per step,
NEVER push. The gate is a **scaling bench** (`scripts/crossthread-roofline-bench`
generalized to drcompiler-emitted kernels): a transform does not count until it
moves measured wall-clock at the chosen thread count, with the cost model's
prediction matched against the measurement.

**Portability is a requirement, not the X3D special case.** Everything
machine-specific is described by a generic **topology model** (§4) sourced from
the cost-model JSON — the same `defaults < JSON < CLI` discipline as the rest of
the cost model (`CROSSCUTTING.md`). The DEFAULT topology is one symmetric cache
domain over the available cores (portable, correct, unspecialized); concrete
parts are JSON profiles. Target classes the model must express, with the dev host
(7950X3D) as only one row:

| machine | cache domains | heterogeneity to exploit (§5.3) |
|---|---|---|
| AMD Zen4 X3D (dev host) | 2 CCX, **asymmetric L3** (96 vs 32 MiB) | asymmetric *cache* — place by working-set fit |
| Intel Xeon SKX/Idun | 1 shared L3 / socket, mesh | uniform; NUMA across sockets |
| AMD EPYC (multi-CCD) | many CCX, symmetric L3 each | many small domains — pack shards per CCX |
| ARM Neoverse (N/V) | shared SLC, per-cluster L2 | NUMA-ish clusters; SVE width |
| ARM big.LITTLE / Apple M | **P+E cores**, shared L2/SLC | asymmetric *cores* — place by compute need |

So §5.3 is "place a shard on the domain whose scarce resource (cache, compute, or
bandwidth) best fits its bottleneck" — the X3D V-cache is one instance of an
asymmetric-cache machine; P/E cores are an asymmetric-compute machine; NUMA is an
asymmetric-bandwidth machine. One placement rule, three machine shapes.

---

## 0. Why this is a cost-model problem, not just a backend

drcompiler's value is the cost model. Parallel codegen here is not "add OpenMP";
it is **the executor for the cost model's parallel decisions**. The split:

- The **cost model decides** (already built / extended here): shard or not (is the
  loop compute-bound enough to scale, or BW-bound and already saturated?), the
  shard count, the grain, the placement (which CCX / node), whether to pad for
  false sharing. Inputs: `ThreadModel`, the roofline (`streamCycles`), the
  per-thread working set, `effectiveCache(L3)`.
- The **codegen executes**: emit the sharded, pinned, padded loop nest + the
  runtime calls.

The two must share one currency. A shard is profitable iff the cost model says
the per-thread work (`WS/N`, the §3.2 partition) is compute-bound at the
per-thread bandwidth (`streamCycles` in EXCLUSIVE mode — one workload owns the
machine, the natural mode for a sharded loop), and the fork/join + barrier cost
is amortized over enough work. That is a `decideShard()` query alongside
`decideBufferStrategy()` / fission.

---

## 1. Where it sits in the pipeline

```
cgeist → affine → dr-opt{ demote, register-block, fission, ... }
       → dr-opt{ NEW: dr-shard }            ← this spec
       → mlir-opt lower (affine→scf→cf→llvm OR →openmp→llvm)
       → mlir-translate → clang → link against libdrpar (the runtime)
```

`dr-shard` runs **after** the affine compute transforms (register-block has
already formed the vectorized micro-kernels and exposed the parallel spatial
loops; fission has made its keep/recompute choices). It shards the outermost
*provably parallel* loop of each hot nest. Running last keeps every existing
analysis (alias, `isLoopParallel`, the register tile) valid and lets sharding
compose with the tile rather than fight it.

Two lowering targets, both kept (pick per `--par-runtime`):
- **`libdrpar` (default)** — a thin drcompiler runtime (§2): `dr-shard` emits
  `func.call @__drpar_for(...)` around an outlined shard body. Full control over
  pinning, first-touch, and the thread pool; no libomp dependency; the runtime is
  ~300 LOC of C.
- **OpenMP dialect** — `dr-shard` emits `omp.parallel`/`omp.wsloop`; lowered by
  upstream `--convert-openmp-to-llvm` against libomp. Portable, less control over
  placement (affinity via `OMP_PROC_BIND`/`OMP_PLACES`), good as a correctness
  oracle and a portability fallback.

---

## 2. The runtime + ABI (`libdrpar`)

A **persistent, pinned thread pool** created once at first parallel region and
reused (fork/join is a barrier on a pool, never a thread spawn). Minimal ABI the
codegen targets:

```c
// Run [lo,hi) sharded across the pool; `body(shardLo, shardHi, tid, ctx)`.
void __drpar_for(int64_t lo, int64_t hi, int64_t grain,
                 void (*body)(int64_t, int64_t, int32_t, void *), void *ctx);
// Topology-aware: allocate `bytes` first-touched by the worker that will own
// shard `tid` (NUMA-local / CCX-local placement). Freed with __drpar_free.
void *__drpar_alloc_local(size_t bytes, int32_t tid);
// Tree/atomic reduction helpers for sharded reductions (§3.3).
void __drpar_reduce_f64(double *perThread, int n, double *out, int kind);
```

Design points:
- **Pinning is the pool's job**, set once at pool creation from the `ThreadModel`
  topology (§4): worker `i` is `pthread_setaffinity_np`'d to a chosen core. The
  codegen never re-pins.
- **Outlining**: `dr-shard` outlines the loop body to a `private` func taking
  `(shardLo, shardHi, tid, ctx)`; `ctx` is a packed struct of the captured
  memrefs + scalars (the same packing register-block already does for its tiles).
- **Grain** = the contiguous block size; `__drpar_for` static-block-partitions
  `[lo,hi)` into `ceil((hi-lo)/N)` blocks by default (§3.2), one per worker — no
  work-stealing in v1 (deterministic placement is worth more than load balance
  for the regular nests drcompiler targets; revisit for ragged/triangular).
- **No exceptions, no allocation on the hot path**; the pool and per-thread
  scratch are allocated once.

The runtime is its own small C TU compiled with the same `clang -O2 -march=native`
and linked by `drcc`; the ABI is the only contract the emitted IR depends on.

---

## 3. Loop sharding (the transform)

### 3.1 What to shard

The **outermost loop of a hot nest for which `affine::isLoopParallel` holds** and
whose trip count is `>= shardCount * minGrain` (else the fork/join is not
amortized). Candidates in drcompiler's world:
- register-block's **outer spatial loop** (`sOut` = oc / batch / the
  unroll-jammed parallel dim) — the compute-bound GEMM/conv kernels. The mr/nr
  jam and the VL micro-kernel are *inside* the shard, untouched.
- the **eltwise/BN/residual nests** (batch × channel × spatial) — BW-bound;
  shard only when the cost model says the per-thread stream is below the BW knee
  (§0), else sharding just multiplies fork/join over a saturated bus.
- direct-conv bands (the `ow` spatial loop) when not already VL-stepped to the
  remainder.

Never shard: a loop carrying a dependence (`!isLoopParallel`), a reduction's
accumulation loop (shard the *spatial* loop around it instead, §3.3), a loop
whose body has unanalyzable aliasing (the existing register-block alias guard is
reused verbatim).

### 3.2 Grain + partition — the per-thread working set becomes REAL

Static **block** partition by default: shard `t` owns `[lo + t*B, lo + (t+1)*B)`,
`B = ceil(trip / shardCount)`. Block (not cyclic) because:
- it gives each thread a **contiguous slice** → spatial locality, hardware
  prefetch, and first-touch NUMA/CCX placement (§5) all work;
- it makes the cost model's `WS/N` exact: the per-thread footprint is one block's
  worth, which is the gap-#2 quantity `CROSSCUTTING.md` could only assume. After
  sharding it is a fact the model can price against `effectiveCache(L3-slice)`.

`shardCount` = `ThreadModel.activeThreads` in EXCLUSIVE mode (own the machine);
in INTERSPERSED mode the cost model picks `< activeThreads` to leave cores for
co-tenants. `minGrain` is set so `B * perIterBytes` is at least a few cache lines
(amortize the prologue) and `B * perIterCost` exceeds the measured fork/join
latency (a one-time §6 calibration constant, ~hundreds of ns on a warm pool).

### 3.3 Reductions

register-block already lowers reductions to `iter_args` (SSA accumulators) or the
dot-family horizontal reduce. To shard a reduction's *enclosing spatial* loop is
trivial (independent outputs). To shard the *reduction dimension itself* (rare —
only when the spatial extent is too small to fill the pool, e.g. a batch-1 GEMV):
emit **per-thread partial accumulators** (`__drpar_alloc_local`, padded §5.1) and
a final `__drpar_reduce_f64` tree-combine. This reassociates the FP sum — gated
on the same `fastmath<reassoc>` the dot family already sets, with the determinism
note in §7.

### 3.4 Nested + collapsed parallelism

v1 shards **one** loop (the outermost parallel one) and leaves inner loops serial
(they become the per-shard kernel — exactly the vectorized register-block tile).
When the outer trip `< shardCount` (batch-1!), **collapse** the outer two parallel
loops into one iteration space before sharding (`affine` supports the index
arithmetic) so the pool stays full. No nested fork/join in v1 (oversubscription +
barrier nesting cost more than it buys on these nests).

---

## 4. Pinned execution + a portable topology model

Pinning is what makes the cost model's cache/BW assumptions *hold* instead of
being averages over a migrating scheduler. The topology is a **generic
machine description** — a set of cores grouped into cache domains and NUMA nodes,
with optional per-domain heterogeneity — not a CCX-specific struct:

```
struct CoreClass { enum { Perf, Eff } kind = Perf; double relThroughput = 1.0; };

struct CacheDomain {                   // one shared cache (an L3 slice / CCX /
  unsigned level;                      //   SLC / a socket's LLC)
  int64_t  sizeBytes;                  // THIS domain's capacity (asymmetric ok)
  double   bytesPerCycle;              // THIS domain's fill bandwidth
  SmallVector<int> cores;              // cores sharing it
};

struct Topology {                      // JSON `thread.topology`; absent => one
  unsigned numCores = 0;               //   symmetric domain over all cores
  SmallVector<CacheDomain> domains;    // >= 1; sizes/BW MAY differ (X3D, P/E SLC)
  SmallVector<CoreClass>   coreClass;  // per-core; uniform Perf by default
  SmallVector<int> coreToNode;         // NUMA node per core (all 0 by default)
  SmallVector<int> workerToCore;       // explicit pin map, else default fill
};
```

- The model is **enumerated, not assumed**: a machine is a list of domains with
  their own size/BW. A symmetric Xeon is one domain; the X3D is two domains with
  different `sizeBytes`; an EPYC is eight equal domains; an Apple M is one cache
  domain but two `CoreClass`es. No part of the codegen names "CCX" — it reasons
  over `CacheDomain`s.
- **Default (no JSON)**: one domain, all cores, `coreClass=Perf`, node 0 —
  portable and correct, just unspecialized. Real parts are JSON profiles shipped
  under `cost-models/` (`zen4-x3d.json`, `skx.json`, `epyc.json`,
  `neoverse-n1.json`, `apple-m.json`).
- **Affinity set once** at pool creation from `workerToCore` (or the default
  compact, domain-packed fill: fill one domain's cores before the next, so a
  shard group fitting one domain stays on it). The cost model's
  `effectiveCache(L3)` becomes the **pinned domain's** `sizeBytes` (not a chip
  average), `streamCycles` its `bytesPerCycle` — both already per-call quantities
  after `CROSSCUTTING`, now indexed by the domain a shard is placed on.
- **Why pinning is load-bearing on EVERY machine**: without it the OS migrates
  threads, any per-domain residency (a V-cache slice, a NUMA node, a P-core) is
  lost, and the roofline's per-thread-BW is a fiction. Pinning is the precondition
  for every §5 specialization — portable, not X3D-only.

---

## 5. The concerns that enable specialized performance

These are the reasons to build this at all — each is a knob the pinned, sharded
form *unlocks* that a serial or unpinned-OpenMP build cannot express.

### 5.1 False-sharing avoidance (the deferred `CROSSCUTTING` P3, now enabled)

Block partition already keeps distinct threads on distinct cache lines *inside* a
slice; the hazard is at **slice boundaries** and in **per-thread reduction
accumulators**. Mitigations the codegen emits:
- align each shard's output base to a cache line and round `B` up to a
  line-multiple where the output is written densely (a few wasted iterations beat
  a ping-ponged line);
- per-thread accumulators (`§3.3`) are **cache-line padded** (`__drpar_alloc_local`
  rounds each thread's region to `cacheLine`). This is the concrete thing the
  cost model's "no false-sharing reasoning" gap asked for.

### 5.2 NUMA-aware placement (forward-looking; single-socket today)

`__drpar_alloc_local(bytes, tid)` **first-touches** the buffer from the worker
that will own shard `tid`, so the pages land on that worker's NUMA node. The
sharded loop then reads/writes node-local memory. The cost model's P4 NUMA tier
(a remote-latency multiplier + per-node BW on `streamCycles`) becomes real input.
Unvalidatable on this host (one node) — the ABI is in place so a multi-socket
target needs only the topology JSON + the cost-model tier, no codegen change.

### 5.3 Heterogeneous-domain placement (portable; one rule, three machine shapes)

When a machine's `CacheDomain`s or `CoreClass`es differ, place a shard on the
domain whose **scarce resource best fits the shard's bottleneck**. One rule,
parameterized by `Topology`, covering:

- **Asymmetric cache** (Zen4 X3D: 96 vs 32 MiB; some Apple-M / future hybrid
  L3s): a shard whose per-thread working set (`§3.2`, now exact) fits the large
  domain but not the small one is routed to the **large-`sizeBytes` domain**.
  `effectiveCache(L3)` is per-domain, so the model sees the win; the topology
  mapping executes it. On a *symmetric* machine every domain is equal and this
  collapses to the compact fill — no special-casing, no X3D assumption baked in.
- **Asymmetric compute** (ARM big.LITTLE, Apple M P/E, Intel P/E): a
  compute-bound shard goes to `Perf` cores; a BW-bound shard (already
  bus-limited, §5.4) can sit on `Eff` cores at lower power without losing
  throughput. `CoreClass.relThroughput` lets the cost model size the shard's
  expected rate per core type.
- **Asymmetric bandwidth** (NUMA, multi-CCD EPYC, multi-socket): a shard's
  stream should land on the domain/node whose `bytesPerCycle` it will use, paired
  with first-touch (§5.2) so the bytes are local.

The decision lives in `decideShard().domain` (§6): pick the domain minimizing the
shard's modelled cost given its `(computeCycles, bytes, WS)` against each
domain's `(sizeBytes, bytesPerCycle, coreClass)`. The X3D V-cache routing falls
out as the asymmetric-cache case; nothing in the rule, the codegen, or the ABI is
X3D-specific — a new part is a new JSON profile. **This is the biggest specialized
win on heterogeneous hardware generally, not a 7950X3D trick.**

### 5.4 Bandwidth partitioning + the shard/serial gate

Sharding a **BW-bound** nest (eltwise, the G4 round-trips) past the bandwidth
knee buys nothing — N threads share one bus. The `decideShard()` query uses the
EXCLUSIVE roofline: shard a nest only while `perThreadBytes / (BW/usedCores)` is
*latency*-bound (more cores still help) and stop adding cores at the knee. For
**compute-bound** nests (the register-block GEMM/conv kernels) scaling is near
linear until the L3/BW slice is exhausted — those are the sharding wins. The cost
model already distinguishes these (the roofline); the codegen consults it for how
many cores to actually use, which can be `< activeThreads`.

### 5.5 Fork/join amortization + region fusion

A persistent pool makes one fork/join a pool barrier (~hundreds of ns warm). Still,
**fuse adjacent parallel regions** sharing the same iteration space + topology
into one `__drpar_for` (e.g. a conv shard immediately followed by its BN/ReLU
shard over the same `[lo,hi)`) so the barrier and the thread-local prologue are
paid once — and the producer's output is consumed by the same pinned thread that
made it (cache-resident hand-off, the parallel analogue of the fission warm
buffer). This is where parallel codegen and the fission cost model meet.

### 5.6 Vectorization × threading compose, not compete

The shard body **is** the existing register-block vectorized micro-kernel (mr × VL
tile). Sharding the outer spatial loop and vectorizing the inner are orthogonal:
per-thread register pressure is unchanged (each thread has its own file — the
`vecRegBudget` is per-core, halved only under SMT, which the topology already
models). No new interaction except the SMT-vec-budget note already in
`CROSSCUTTING` P0.

---

## 6. Cost-model integration: `decideShard()`

A new query, same currency as fission/DR:

```
struct ShardPlan { unsigned cores; int64_t grain; int domain; bool padBoundary; };
ShardPlan decideShard(loopTrip, perIterComputeCycles, perIterBytes,
                      const MachineModel &mm);   // mm carries the Topology
```

- `cores`: largest `c <= activeThreads` (EXCLUSIVE) for which the per-thread work
  is still latency-bound by the roofline AND `trip/c >= minGrain`. BW-bound nests
  cap early (§5.4).
- `grain`: `ceil(trip / cores)`, rounded to a cache line for dense outputs (§5.1).
- `domain`: the `CacheDomain` (and, on a hybrid part, the `CoreClass`) minimizing
  the shard's modelled cost over the topology — the per-domain `sizeBytes` /
  `bytesPerCycle` / core type (§5.3). On a single-domain machine this is always 0
  (the portable default); the X3D V-cache, P/E, and NUMA choices are the same
  query over different topologies.
- `padBoundary`: set when distinct shards write within a cache line of each other.

The query is pure (no IR), **topology-driven, no machine baked in**; `dr-shard`
reads it and emits accordingly. `MachineModel` already carries the cross-thread
model (`ThreadModel`, `streamCycles`, `effectiveCache`); §4 adds the `Topology`,
so a new target is a JSON file, never a code change.

---

## 7. Correctness + determinism

- **Only provably-parallel loops are sharded** (`isLoopParallel` + the inherited
  alias guards). A missed-parallel loop stays serial — a perf miss, never a
  correctness bug.
- **Reductions reassociate** when the reduction dim itself is sharded (§3.3),
  identical to the dot-family `fastmath<reassoc>` already in use; the end-to-end
  bench's `norm-rel-err <= 1e-4` gate covers it (FP results differ in the low bits,
  top-1 unchanged — the ONNX campaign already accepts this). A `--par-deterministic`
  flag forces a fixed-order tree combine (slower) when bit-stability is required.
- **No data races by construction**: distinct shards own disjoint output slices;
  shared *reads* (weights, the source) are fine; the only writes that touch a
  shared line are the padded reductions (§5.1).
- **The OpenMP lowering is the oracle**: emit the same shard via `omp.parallel`
  and diff results against `libdrpar` to catch outlining/packing bugs.

---

## 8. Phasing + gates

Each phase: one commit, gated on the scaling bench, honest verdict, NEVER push.
Default OFF (`--par-shard=0`) so every existing serial path is byte-identical.

- **P0 — runtime + minimal shard.** `libdrpar` (pool + `__drpar_for`, compact
  pinning) + `dr-shard` shards the outermost parallel `sOut` of register-blocked
  GEMM/conv into `__drpar_for` over the static blocks. No NUMA/V-cache/padding.
  Gate: a sharded PolyBench GEMM scales (measure 1..16 cores; near-linear until
  the L3 slice) and matches the OpenMP oracle bit-for-bit (no reassoc yet).
- **P1 — `decideShard()` + the BW gate.** Wire the cost model so BW-bound nests
  cap cores at the knee and compute-bound scale; validate the predicted `cores`
  against the measured scaling knee (the crossthread bench, drcompiler-emitted).
- **P2 — heterogeneous-domain placement (§5.3).** Per-domain `effectiveCache` +
  topology routing, driven by the JSON profile. Gate is per machine class and the
  model must predict each: on Zen4-X3D, a WS-in-(small,large]-domain kernel is
  faster pinned to the large-cache domain than compact-filled; on a hybrid
  P/E part, a compute-bound shard on `Perf` cores beats a naive round-robin; on a
  symmetric part the plan is identical to P0 (no regression). One code path, the
  JSON selects the win.
- **P3 — region fusion + warm hand-off (§5.5)** and **false-sharing padding
  (§5.1)**; gate: a conv→BN→ReLU chain fused into one pinned region beats three
  separate ones; padded reductions beat unpadded on a sharded GEMV.
- **P4 — NUMA (§5.2)** when a multi-socket target exists; ABI already present.
- **resnet50 end-to-end**: the existing `onnx-codegen-bench.sh` extended with a
  `--par N` config; the honest bar is **EXCLUSIVE-mode single-inference latency**
  (batch 1, all cores, pinned) vs onnx-mlir --O3 with the same thread count, and
  the cross-thread cost model's mode set to `exclusive`.

## 9. Risks + explicitly OUT

- **Batch-1 has little parallelism** — the spatial/channel dims are the only
  slack; §3.4 collapse is mandatory, and some layers (7×7 at 49 spatial) will not
  fill 16 cores. The honest expectation: exclusive-mode latency improves on the
  big early layers, flattens on the small late ones; the cost model should *not*
  shard where the grain is below `minGrain` (a measured floor, not a guess).
- **Work-stealing / dynamic grain** — OUT in v1 (deterministic placement >
  balance for regular nests); revisit only if triangular/ragged kernels (the
  PolyBench peels) show idle cores.
- **Threading the whole pipeline** (parallel `dr-opt` itself) — OUT; this is about
  the *emitted* code.
- **GPU / offload** — OUT.
- **Auto-detecting the deployment mode** — OUT; `exclusive`/interspersed stays a
  JSON property (whole-program), per `CROSSCUTTING` III.4a.

## 10. Mechanics (carry-overs that will bite)

- The shard body packing reuses register-block's ctx-struct outlining; do not
  invent a second packer.
- Pin BEFORE first-touch, or the pages land wrong (allocate inside the pinned
  worker, not on the main thread).
- `clang -O2 -march=native` must compile `libdrpar` with the same flags as the
  kernel TU, or the ABI struct layout can skew (pack `ctx` explicitly).
- The bench must pin its own threads the same way, or the comparison measures the
  scheduler, not the code (the residency lesson from the crossthread bench,
  general to any per-domain effect: an unpinned run hides it — on the X3D the
  V-cache CCD, on a NUMA box the local node, on a hybrid the P-cores).
- Medians, back-to-back, FLOP-counted before believing a scaling claim — the same
  discipline as the serial campaign.
