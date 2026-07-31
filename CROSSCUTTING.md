# CROSSCUTTING — cost-model unification analysis + a preliminary cross-thread spec

Status: ANALYSIS + PRELIMINARY SPEC (2026-06-12). Scope: the *whole* cost model
(`Analysis/{MachineModel,CpuCostModel,ArchHandler,RegisterPressureAnalysis,ReuseAnalysis}`)
and every consumer (register-block, memory-fission, data-recomputation, the
loop-tiling/-fusion/-distribution forks, stencil-time-tile). Method: a 5-lens
source audit (cache / register / vectorization / prediction / parallel), every
claim tied to `file:line`, cross-checked against the campaign docs
(`claude-docs/COSTMODEL_PORTABILITY_FINDINGS.md`, `CONTENTION_AWARE_COSTMODEL.md`,
`COSTMODEL_SPIKE_FINDINGS.md`). Companion to `ONNX_O3_GAP_RESULTS.md`.

**Headline.** The cost model is *partially* unified. `MachineModel` (the v4
single-source-of-truth) and `dr::estimateLoadLatency` (the one cycle-pricing
primitive) genuinely unify the cache/contention currency for the three
"good" consumers (DR, fission, register-block, stencil). But there are **two
non-communicating register models**, **three disconnected notions of vector
width**, **two contention-blind cache forks** (loop-tiling, loop-fusion), and the
declared canonical helpers (`effectiveLLC()`, `tlbReachBytes()`) have **zero
callers**. For parallelism the model has exactly **one** thread-aware quantity
(`llcSharers`) and **no bandwidth/roofline term at all** — a gap the code itself
already names ("the still-missing roofline currency", `MemoryFission.cpp:405`).
It is **not ready for parallel workloads**; Part III specifies what it needs.

---

# Part I — How unified is the cost model?

## I.1 The four cost-model objects

| object | owns | JSON key namespace | resolution |
|---|---|---|---|
| `MachineModel` (`Analysis/MachineModel.h:31`) | cache geometry (l1/l2/l3/line/latencies), `llcSharers`, page/TLB, **vector-exec model** (`vectorBitsNative/Arch`, `vecRegBudget`, `avx512FreqThrottle`) | `cache.*`, `arch.vector_bits_*`/`vec_reg_budget`/`avx512_freq_throttle` | `fromJson` + `defaults < JSON < explicit-CLI hasValue()` (`.h:11-16`) |
| `dr::CacheParams` (`DataRecomputation/CacheCostModel.h:43`) | the POD cache struct the cost *helpers* actually read (+ `l2OccupancyPct`) | — (copied field-by-field from resolved options) | per-call-site construction |
| `ArchHandler` / `ArchParams` / `RegisterParams` (`Analysis/ArchHandler.h`, `RegisterClass.h:64`) | per-arch register budgets (gp/fp/vec/pred), `vectorWidthBits`, combine weights α/β/γ, spill costs | `arch.{vector_width_bits,handler,weights}`, `registers.*` | `ArchHandler::create(name)` + JSON override |
| `CpuCostModel` op table (`Analysis/CpuCostModel.cpp:21`) | per-op cycle costs (`mulf=3`, `div=15`, `sqrt=20`, default 5) | `ops.*`, `default_cost` | `loadFromFile` |

Four objects, **two of which (`MachineModel` vs `dr::CacheParams`) describe the
same cache** and are kept in sync only by hand-copying at each consumer.

## I.2 Cache — one source, two types, two forks

- **Unified path (good):** DataRecomputation (`DataRecomputation.cpp:1394`),
  MemoryFission (`MemoryFission.cpp:253-268`), AffineRegisterBlock
  (`AffineRegisterBlock.cpp:766`), AffineStencilTimeTile (`:176`) all resolve
  geometry via `MachineModel::fromJson` and then call the single tiered pricer
  `dr::estimateLoadLatency` (`CacheCostModel.cpp:144`) — bytes → l1/l2·occ/effL3/mem
  latency. Contention (`effL3 = l3Size/llcSharers`) reaches DR + fission *for
  free* through this one function, exactly as `CONTENTION_AWARE_COSTMODEL.md`
  intended.
- **Fork #1 — LoopTiling** (`AffineLoopTilingCostModel/LoopTiling.cpp:247`) reads
  `l2Size` *straight from `CpuCostModel::cacheParams()` JSON*, bypassing
  `MachineModel`; a CLI `--l2-size` never reaches it. Acceptable today only
  because it tiles to private-L2/2 (no `llcSharers` needed).
- **Fork #2 — LoopFusion** (`AffineLoopFusionCostModel/LoopFusion.cpp:83`) is a
  hardcoded `dr::CacheParams{32768,1048576,33554432,4,12,40,200,64}` literal,
  patched by a partial JSON merge (`:2229`) that folds l1/l2/l3+latencies but
  **omits `cacheLineSize` and `llcSharers`** → it can never become
  line-size-portable or contention-aware. (The old *stale* `262144`/`l3=0`
  literal is FIXED in source; the portability doc still lists it as live — that
  drift is now doc-vs-code, not a runtime defect.)
- **Dead canonical API:** `MachineModel::effectiveLLC()` (`.h:102`) and
  `tlbReachBytes()` (`.h:109`) have **zero callers** — the `l3/sharers` derate is
  open-coded at 6+ sites (`CacheCostModel.cpp:158`, `MemoryFission.cpp:395`,
  `AffineRegisterBlock.cpp:874,1024`, `AffineStencilTimeTile.cpp:255,272,514`).
  They agree numerically *today*; the risk is structural (6 copies, one pinned to
  sharers=1).
- **Residual hardcodes:** MemoryFission literals `mem=200, line=64` into its
  `CacheParams` (`MemoryFission.cpp:396`) despite `mm.memLat`/`mm.cacheLine`
  carrying JSON values; `ReuseAnalysis` hardcodes `kCacheLineBytes=64` twice
  (`ReuseAnalysis.cpp:129,145`) so the whole tiler/fusion reuse-distance path is
  line-size-non-portable; `AffineStencilTimeTile.cpp:273` bakes an f64 element
  size (`box = effLLC/2/16`).
- **Two cache *currencies*:** the tiler/fusion/distribution ride `ReuseAnalysis`
  (exact byte counts vs a raw `cacheBytes`, `loopCarriesEvictedReuse`), while
  DR/fission/register-block ride `estimateLoadLatency` (bytes → latency tiers).
  These never cross-check.

## I.3 Register pressure — TWO parallel, non-communicating models

1. **PRESSURE model:** `ArchHandler::defaultRegisters() → RegisterParams{gp/fp/vec/pred}`
   consumed by `RegisterPressureAnalysis` (liveness + `SpillStrategies/{ExcessHot,SumExcess,GraphColor}`).
   JSON key `registers.vec_budget` (default 16; AVX-512 handler 32). Consumers:
   DataRecomputation/BufferElim, the LoopFusion fork, PrintRegisterPressure.
2. **TILE model:** `MachineModel.vecRegBudget` (default **24**) +
   `vectorBitsNative/Arch`, consumed *only* by AffineRegisterBlock's
   `preferredVectorElems` to pick `vl`. JSON key `arch.vec_reg_budget`.

These are sourced from **different JSON keys**, have **different defaults
(16 vs 24 vs the handler's 32)**, and **never reconcile** — `MachineModel::fromJson`
reads only `cache`+`arch`, and `RegisterPressureAnalysis` never sees
`vecRegBudget`. **The pass that actually materializes the `mr·⌈nr/vl⌉` live
vector accumulators (AffineRegisterBlock) never invokes
`RegisterPressureAnalysis`** (grep-empty): its entire register reasoning is the
single inequality `mr·⌈nr/vl⌉ ≤ vecRegBudget` inside `preferredVectorElems`, and
even that runs only when a JSON sets `hasExplicitVectorModel`. The
stream+broadcast registers the kernel keeps live are not counted (they are baked
into the `24 = 32−8` constant; if `mr`/`nr` change, the reservation does not).
The FP/Vec class split compounds this: `RegisterPressureAnalysis` analyzes the
*pre-vectorization* (FP-class) body; after register-block+vectorize the values are
Vec-class and **never re-analyzed**.

## I.4 Vectorization — one cost-modeled decision (gated off), the rest structural

- **Whether to vectorize / which family / which kernel:** *not cost-modeled.*
  `vectorize=true` is always-on (`Passes.td:654`); `detectFamily` is a pure
  stride classification (`Vectorize.cpp:73`); the micro-kernels include no
  cost-model header and run on legality alone. There is **no scalar-vs-vector
  veto** — on a hypothetical narrow-SIMD / high-throttle target where
  vectorizing loses, nothing stops it.
- **Which VL:** cost-modeled via `preferredVectorElems`, but **JSON-gated**
  (`hasExplicitVectorModel`) → on the default machine `vl` stays the measured
  constant 8 and the *only predictive piece is dead code*.
- **Three vector-width notions, three JSON keys, no cross-check:**
  (a) `MachineModel.vectorBitsNative/Arch` (`arch.vector_bits_*`) drives the VL
  element count; (b) `ArchParams.vectorWidthBits` (`arch.vector_width_bits`)
  drives register-pressure `classifyType` (`X86_64_AVX512.cpp:44`); (c) the JSON
  carrier for (b). A JSON can set `vector_width_bits=512` (pressure thinks
  zmm) while `vector_bits_native` is absent (register-block still emits vl=8
  ymm) — silently inconsistent.
- **`avx512FreqThrottle` is fully inert** — parsed into `MachineModel`, read by
  no decision anywhere. Its own comment says it "feeds cross-machine THROUGHPUT
  prediction", but no throughput consumer exists. (Part III.6 gives it a job.)

## I.5 Prediction fidelity — a small validated island in a large heuristic sea

| mechanism | kind | validated? |
|---|---|---|
| `estimateLoadLatency` tiered residency (`CacheCostModel.cpp:144`) | predictive (bytes→cycles) | yes — `bytesToMemCycles` lines×latency fix validated vs gemver (`LoopFusion.cpp:93`) |
| contention reuse-distance (`effL3=l3/sharers`) | predictive | **yes** — 5-WS × 2-contention sweep on 7950X3D; WS4.6M SKIP→FISSION reversal (2.58× measured), pinned by `test/MemoryFission/fission-contention-aware.mlir` (`CONTENTION_AWARE_COSTMODEL.md:104`) |
| `preferredVectorElems` VL | predictive | **yes** — Spike #4 (f32 N=1024: vl8≈88 > vl16≈77 > vl4≈55 GFLOPs) + resnet50 vl16-vs-8 −0.008 s (`COSTMODEL_PORTABILITY_FINDINGS.md`) — but gated off at default |
| stride-aware spatial `min(line, stride·elem)` (`CacheCostModel.cpp:584`) | predictive | yes — line 64→128 moves a gather footprint 262148→393220 |
| 8×16 mr/nr tile | heuristic (measured constant) | "within 5% of optimum" (`COSTMODEL_SPIKE_FINDINGS.md`) but never recomputed per-machine |
| combine weights α/β/γ | **dead** | all default 1.0; **no handler overrides them** → `combineCosts` is a plain unweighted sum; on the DR path β·reg and leafLoadCost are fed **0** |
| op-cost table (`mulf=3`,`div=15`) | heuristic, uncalibrated | **no** — asserted x86 latencies, a *second* currency never cross-checked against the latency tiers |
| `kDefaultTripCount=128`, `kIssueWidth=4` | heuristic, forked-hardcoded | `128` under-counts a 10⁶-element stream → flips keep/recompute to the optimistic-WRONG side |

So the model **predicts** in three validated places (cache residency, contention
reversal, VL) and is otherwise a set of tuned thresholds. The decision surface is
a **4-value step function** (the latency tiers): dead-flat within a tier,
hypersensitive at boundaries — the contention doc notes α had to reach **100×**
to move any decision.

## I.6 Unification scorecard

| consumer | cache source | register source | combiner | predictive? |
|---|---|---|---|---|
| DataRecomputation | MachineModel → `dr::CacheParams` | RegisterPressureAnalysis (but reg fed 0) | `ArchHandler::combineCosts` (unweighted) | partly (residency) |
| MemoryFission | MachineModel (but mem/line literal) | — | hand-rolled min-of-costs | **yes** (contention) |
| AffineRegisterBlock | MachineModel (effLLC open-coded) | `MachineModel.vecRegBudget` only | hand-rolled WS≤cache | partly (VL, gated) |
| AffineStencilTimeTile | MachineModel (f64 baked) | — | hand-rolled | heuristic |
| LoopTiling | **CpuCostModel JSON** (fork) | — | ReuseAnalysis | predictive (reuse) |
| LoopFusion | **hardcoded literal** (fork) | RegisterParams | ReuseAnalysis + `bytesToMemCycles` | partly |
| LoopDistribute | flat 512 KiB option | — | ReuseAnalysis | heuristic |

---

# Part II — Parallel readiness

## II.1 The one thread-aware mechanism (and its real win)

`MachineModel::llcSharers` / `effectiveLLC() = l3Size/llcSharers`
(`MachineModel.h:38,102`) is the *entire* parallel surface. Its input is unified
(one parse path, `CpuCostModel.cpp:148 → MachineModel.cpp:40`); its formula is
duplicated inline at ~6 sites but agrees. It is **validated**: the WS4.6M
SKIP→FISSION reversal as sharers rise is measured and pinned (II.5 above). Credit
where due — this is a genuine contention-aware win for the consumers that were
tuned (DR, fission).

But it is a **static co-tenant count**, not a thread/work model. Everything below
it is single-thread.

## II.2 The gaps (ranked by impact on a parallel workload)

1. **No bandwidth / roofline term — anywhere.** Every "contention" decision is a
   *capacity* derate feeding a *latency* tier. Under N threads the binding
   constraint is shared **DRAM/LLC bandwidth**, which is invisible: a fission
   buffer round-trip and a recompute that streams the source are both priced by
   per-access latency, never by bytes/s contended across cores. The code already
   admits this — `MemoryFission.cpp:405`: *"A tier-latency cost cannot express
   that warm-vs-evicted asymmetry without a per-level BANDWIDTH term (the
   still-missing roofline currency)."* This is **the** gap.
2. **`effectiveLLC` double-counts for a parallelized loop.** `l3/sharers`
   assumes N *independent* co-tenants each with a *full* footprint. But for one
   loop parallelized across N threads (one shared problem), each thread's working
   set is footprint/N, not the full footprint against l3/N. The model charges the
   full single-thread WS against the derated cache — a **double penalty**
   (`MachineModel.h:102` semantics vs `CacheCostModel.cpp:219` full-extent trips).
3. **No work-partition model.** Trip counts are full loop extents
   (`CacheCostModel.cpp:219`); the model never reasons that a `parallel-for`
   splits the outer dim → per-thread WS and reuse distance are wrong.
4. **SMT not modeled.** L1/L2 are *asserted* private and never derated
   (`CacheCostModel.cpp:147`); an SMT sibling shares them. `vecRegBudget` is
   per-core but the physical vector file is split across SMT threads. Only the
   flat `l2OccupancyPct` knob partially absorbs this.
5. **`avx512FreqThrottle` is load-dependent *and* dead.** All-core sustained
   512-bit FMA downclock is a *shared* effect whose magnitude grows with the
   number of cores running 512-bit kernels — exactly a cross-thread quantity. The
   field exists (from WP-G1) but feeds nothing (`MachineModel.h:82`).
6. **False sharing on the fission output buffer.** Fission materializes a fresh
   `memref<?xT>` with no alignment (`MemoryFission.cpp:394`); N threads writing
   adjacent slices ping-pong cache lines. Unmodeled.
7. **No NUMA.** One scalar `l3Size`/`memLat` = one memory domain; remote-LLC and
   remote-DRAM (≈1.5–2× latency, per-node bandwidth) are inexpressible.
8. **Three contention-blind forks.** LoopTiling (private-L2/2), LoopDistribute
   (flat 512 KiB), LoopFusion (sharers pinned 1) never receive `llcSharers`; on a
   loaded box they mis-decide while DR/fission/register-block adapt.

---

# Part III — Preliminary spec: cross-thread cost model

## III.0 Goals and invariants

- **One thread/topology model**, owned by `MachineModel`, consumed uniformly —
  the same discipline the v4 cache unification aimed for, finished and extended.
- **`nThreads = 1` is byte-identical.** Every new term degenerates to today's
  numbers at one thread / infinite bandwidth, gated like the WP-G1 vector model
  (`hasExplicitThreadModel`) so the whole existing lit suite and PolyBench stay
  identical. This is the hard acceptance gate.
- **JSON-driven, measurement-gated.** Bandwidth/NUMA numbers are machine-specific
  → cost-model JSON; every new decision lands behind a measured spike (the
  campaign's spike-first rule), pinned by a contention-style lit test.

## III.1 Pre-req: finish the single-thread unification (P0, do first)

A cross-thread layer on top of a forked single-thread model just multiplies the
forks. Split into byte-identical *refactors* (landed) and *design changes* that
are NOT byte-identical and belong in the implementation phases.

**Landed — byte-identical refactors (lit 218/0 throughout):**
- ✅ `costmodel_p0a` — route all 6 open-coded `l3/sharers` sites through the new
  static `MachineModel::effectiveLLC(l3,sharers)` (the dead canonical API now has
  one definition and all callers; the single place to make the share per-thread).
- ✅ `costmodel_p0d` — fission sources `memLatency`/`cacheLine` from `MachineModel`
  instead of the hardcoded `200`/`64` literals.
- ✅ `costmodel_p0e` — thread `cacheLine` through `ReuseAnalysis`
  (`loopCarriesEvictedReuse`/`anyTemporalReuse` gain a `cacheLineBytes` param,
  default 64); the tiler passes the JSON-resolved line.
- ✅ `costmodel_p0f` — the LoopFusion fork's JSON merge now folds `cacheLineSize`
  and `llcSharers` (was line-non-portable + contention-blind, sharers pinned 1).
- ✅ `costmodel_p0g` — `kIssueWidth = 4` (the last hardcoded core-µarch constant
  on the DR decision path, `CacheCostModel.cpp`) becomes
  `ArchParams::issueWidth`: per-handler default, JSON `arch.issue_width`,
  threaded to all three `estimateComputeCost` call sites from ArchParams already
  in scope. Byte-identical — every validated handler (generic / avx2 / avx512 /
  neon) defaults to 4. Pinned by
  `test/DataRecomputation/issue-width-throughput-floor.mlir` (one wide-shallow
  cone priced 167 / 42 / 23 at width 1 / 4 / 8).
- ✅ `costmodel_p0h` — fission gains the `mem-latency` / `cache-line-size` CLI
  overrides it lacked, so its geometry follows the full CLI > JSON > default
  contract rather than JSON > default. Pinned by
  `test/MemoryFission/fission-mem-latency-resolution.mlir` (all four tiers,
  including CLI-beats-JSON).
- ✅ `costmodel_p0i` — two handlers behind the existing `ArchHandler` interface:
  `apple-m-series` (NEON 128 b, 31/32/32/0, **issue 8** — the first non-4 part)
  and `arm-sve` (VL floor 128 b, pred budget 16). `pickHandlerForTriple` routes
  aarch64-darwin to apple-m-series; SVE is name-only (a triple never implies the
  feature). Apple's 128 B line stays *memory* geometry — supply it as
  `cache.cache_line: 128` in a probed JSON, not from the handler.

**Deferred — NOT byte-identical; these are design changes, do in P1/P2:**
- **Register-budget merge** (`vecRegBudget=24` vs `RegisterParams.vecBudget=16/32`)
  is NOT a rename: they are *different concepts* — `vecRegBudget` is
  "accumulator slots usable by the tile" (= total − ~8 stream/broadcast reserve),
  `vecBudget` is the *full* vector file for spill counting. The right unification
  is to **derive** `vecRegBudget = archVecBudget − reserve` from one source, which
  couples `MachineModel` to `ArchHandler`/`RegisterParams` and *changes* the VL
  decision under a JSON that sets `registers.vec_budget`. → P1 (with the
  SMT-split, which is the cross-thread reason to touch it).
- **register-block ↔ `RegisterPressureAnalysis`.** Making register-block *consult*
  the liveness/graph-coloring pressure model (instead of the lone
  `mr·⌈nr/vl⌉ ≤ vecRegBudget` inequality) can change tile/VL decisions → not a
  refactor. P2; a byte-identical first step is a *diagnostic-only* cross-check.
- **Vector-width merge** (`vectorBitsArch` vs `ArchParams.vectorWidthBits`): same
  physical quantity, different defaults (512 vs 128-generic) and different
  consumers; collapsing them changes register-pressure counts on the generic
  arch. → P1, behind the gate.
- **`avx512FreqThrottle`**: not dead (it triggers `hasExplicitVectorModel`) but
  feeds no *decision*. Wiring it is the all-core-throttle of III.6 → P2 (a
  behavior change by design).

## III.2 The `ThreadModel` (new `MachineModel` section)

```
struct ThreadModel {                       // all default to a 1-thread no-op
  unsigned activeThreads      = 1;         // parallel workers actually running
  unsigned coresPerLLC        = 1;         // = today's llcSharers, but DERIVED
  unsigned smtPerCore         = 1;         // SMT siblings sharing L1/L2
  // Roofline currency (III.3). 0 / absent => bandwidth term disabled (no-op).
  double   dramBytesPerCycle  = 0.0;       // sustained 1-thread streaming BW
  double   llcBytesPerCycle   = 0.0;       // shared-LLC fill BW
  // Cache sharing topology (today's asserted L1/L2-private, L3-shared default).
  enum Level { L1, L2, L3 };
  bool     shared[3]          = {false,false,true};
  unsigned numaNodes          = 1;         // >1 => remote tier (III.7, later)
  bool     hasExplicitThreadModel = false; // gate: false => byte-identical
};
```

Derived helpers (replace the open-coded divisions, give the parallel-correct
ones a single home):

```
// Per-thread share of a cache level under the current topology.
int64_t effectiveCache(Level lv) const {
  int64_t cap = sizeOf(lv);
  if (thread.shared[lv]) cap /= max(1u, thread.coresPerLLC);   // LLC contention
  else                   cap /= max(1u, thread.smtPerCore);    // SMT-private split
  return cap;
}
// effectiveLLC() becomes effectiveCache(L3) — one definition, all callers.
```

`llcSharers` becomes a derived view (`coresPerLLC`) so existing JSON keeps
working; `coresPerLLC` defaults to it when the thread block is absent.

## III.3 The roofline currency (the missing axis)

Add a **bandwidth-bound** alternative to every latency estimate. The cost of
moving `bytes` is the *max* of the latency-bound and bandwidth-bound times:

```
// cycles to move `bytes` from `lv`, under the current thread topology.
double moveCycles(int64_t bytes, Level lv) const {
  double lat = latencyBound(bytes, lv);                  // today's tiered model
  double bw  = thread.bandwidthFor(lv);                  // bytes/cycle for this level
  if (bw <= 0) return lat;                               // disabled => no-op
  double perThreadBW = bw / max(1u, thread.activeThreads); // ROOFLINE: shared BW split
  return std::max(lat, bytes / perThreadBW);
}
```

This single change makes memory-bound kernels visible: at `activeThreads=1` it is
`max(lat, bytes/BW)` (a roofline even single-thread — strictly better than
today's latency-only); at N threads the per-thread BW collapses to `BW/N` and
streaming costs scale, which is what actually happens. `estimateLoadLatency`
becomes the latency arm of `moveCycles`. This is the term the fission comment
asks for, and it is what turns the `sourceThrashes` *special clause* into a
*derived* result (warm buffer = capacity-resident, evicted source = BW-bound).

## III.4 `ParallelContext` — work partitioning threaded through queries

The cost helpers must know whether the loop under analysis is data-parallel and
across how many threads. A small context, passed alongside `CacheParams`:

```
struct ParallelContext {
  unsigned threads = 1;        // workers this loop is split across (1 => serial)
  bool     isParallelLoop = false;   // outer dim partitioned => per-thread WS = WS/threads
};
```

Footprint/reuse-distance queries take it and divide the partitioned extent:
`estimateInterveningFootprint`, `reuseDistanceBytes`, the register-block
cache-tile WS, and fission's `perConsumerFP` all compute **per-thread** working
sets (`WS/threads` for a parallel loop) and compare against
`effectiveCache(L3)` — fixing the gap-#2 double-count. At `threads=1` this is a
no-op.

### III.4a — the two workload interpretations (decided by the P1 bench)

Building + measuring the roofline (`scripts/crossthread-roofline-bench`) exposed
a fork that the spec above glossed: `activeThreads` means two different things,
and the cost is different in each.

1. **Independent tenants** (SPEC-rate, batch inference, N processes): each of N
   threads runs the WHOLE problem; the only coupling is bandwidth contention.
   Per-thread bytes = `WS` (full), per-thread BW = `BW/N` →
   **time = `WS / (BW/N)` = `WS·N/BW`** (the cost rises with load). Capacity:
   `WS` (full) vs `effectiveCache = L3/N`.
2. **Parallelized loop** (one inference split across cores): each thread does a
   SLICE; per-thread bytes = `WS/N`, per-thread BW = `BW/N` → the divisors
   **cancel**: **time = `WS/BW`** (independent of N). Capacity: `WS/N` vs `L3/N`
   = `WS` vs `L3` (the sharing cancels — no net derate).

**What is implemented + validated today (P1a/P1b): interpretation #1.**
`streamCycles(bytes, …) = bytes / (BW/activeThreads)` is the tenants model, and
the bench measures exactly that (independent per-thread problems → per-problem
time stays high under load). So `activeThreads` in the shipped code is really
"bandwidth co-tenants", the BW analogue of `llcSharers` for capacity — coherent
and measured.

**RESOLVED — DUAL MODE (`costmodel_p2dual`).** Both interpretations ship,
selected by `thread.exclusive` in the cost-model JSON; `streamCycles` differs
only in the bandwidth divisor:

```
effBW = exclusive ? bw : bw / activeThreads;   // owns vs shares the bandwidth
streamCycles = bytes / effBW;
```

- `exclusive: false` (default) = **interspersed**: `bytes·N/BW` (shared BW;
  pair with `llc_sharers=N`). The P1 behaviour, byte-identical, bench #1.
- `exclusive: true` = **exclusive**: `bytes/BW` (owns BW; pair with
  `llc_sharers=1`). The single-big-parallel-job deployment.

The same 2 MiB buffer's keep cost is then `load=40` serial / `1e6` interspersed
/ `131072` exclusive (16× cheaper than interspersed) — pinned by
`recompute-roofline-bandwidth.mlir`. And the dual model is **empirically
justified**: the extended bench (`...-bench` now measures BOTH — N independent
problems vs ONE problem split across N cores) shows that at compute intensity
fops≈4–16 the optimal **DIFFERS by mode** — interspersed → RECOMPUTE, exclusive →
MATERIALIZE (same kernel, opposite decision). A single-mode model gets one wrong;
the dual model gets both right.

## III.5 Per-consumer cross-thread analysis (the deliverable)

| consumer | single-thread assumption (file:line) | cross-thread query it needs |
|---|---|---|
| **MemoryFission** | full WS vs `l3/sharers`; latency-only; unaligned buffer (`MemoryFission.cpp:394`) | per-thread WS = WS/threads vs `effectiveCache(L3)`; **`moveCycles` (BW)** for buffer round-trip vs source re-stream — the decision *flips* under BW pressure (recompute streams source N× = BW-bound; keep = one shared buffer); **pad the output buffer to a cache line** when `isParallelLoop` (false-sharing fix) |
| **DataRecomputation** | keep = N·loadLat, recompute = N·ALU, reg fed 0 (`CacheCostModel.cpp:167`) | keep priced with **`moveCycles`** (shared BW), recompute ALU **scales per-thread** → under BW saturation recompute wins *more* as threads rise; this is a real, currently-invisible cross-thread reversal |
| **AffineRegisterBlock** | cache-tile WS vs `effLLC`; `vl` ignores throttle (`AffineRegisterBlock.cpp:1024`) | cache-tile against `effectiveCache(L2/L3)` with SMT split; `vl` consults the **all-core throttle** (III.6); register file unchanged (per-thread) but **halve `vecBudget` under SMT** if the sibling is active |
| **LoopTiling / LoopFusion / LoopDistribute** | contention-blind (private-L2/2, literal, flat 512 KiB) | route `cacheBytes` through `effectiveCache(level)`; tile target shrinks under contention; **BW-aware tile** (a tile that fits cache but saturates BW is no win) |
| **AffineStencilTimeTile** | `box = effLLC/2/16` (f64, 1-thread) | per-thread time-tile box vs `effectiveCache(L3)`; the skew box is per-thread under a parallel time loop |
| **RegisterPressureAnalysis** | spill slot = L1-resident 5cy; remat priced once (`RegisterPressureAnalysis.cpp:176`) | spill-reload via `moveCycles` (a contended spill slot is not L1-resident); remat pressure ×threads if the cloned region is replicated per worker |

## III.6 Wiring `avx512FreqThrottle` to the thread model

The all-core AVX-512 license downclock is the cleanest cross-thread/vectorization
coupling and it reuses an existing (currently dead) field. Model the throttle as
a function of how many cores run wide FMAs:

```
double effFreqScale() const {                 // 1.0 at light load
  if (thread.activeThreads <= 1) return 1.0;
  // interpolate base..avx512FreqThrottle by active-core fraction
  double f = double(thread.activeThreads) / max(1u, thread.coresPerLLC);
  return 1.0 - (1.0 - avx512FreqThrottle) * min(1.0, f);
}
```

Then the VL decision and the roofline both scale wide-FMA throughput by
`effFreqScale()`. On a native-512 Xeon this is where vl=16 could *lose to* vl=8
at all-core (severe throttle) even though it wins single-thread — the exact
question the planned **Idun (56-core Xeon Gold) spike** must answer. The spike
becomes: measure the all-core 512-bit downclock, set `avx512_freq_throttle` +
`active_threads` in `idun-xeon.json`, and check whether the model's VL flips.

## III.7 Phasing, validation, risks

**Phasing** (each phase byte-identical at `nThreads=1`, gated on
`hasExplicitThreadModel`):
- ✅ **P0** — the III.1 single-thread unification cleanups (no behavior change).
  Landed: `costmodel_p0a/d/e/f/g/h/i`.
- ✅ **P1** — `ThreadModel` fields + `effectiveCache()` + `streamCycles()`
  bandwidth term, wired into fission (`costmodel_p1a`) and DR (`costmodel_p1b`),
  the memory-bound consumers. Gated, byte-identical at default. Validated at the
  *decision* level by `test/MemoryFission/fission-roofline-bandwidth.mlir` (the
  clean keep→recompute flip) and the DR cost mechanism test. NOTE: the roofline
  is currently *additive* to fission's `sourceThrashes` clause, not yet a
  replacement — fully deriving `sourceThrashes` from the bandwidth term (so it
  stops being a separate special-case) needs the recompute side's source re-read
  priced in bytes too, a P1.5 follow-up. EMPIRICAL validation: **DONE on this
  host** — `scripts/crossthread-roofline-bench.{c,sh}` measures the
  MATERIALIZE-vs-RECOMPUTE crossover directly (OpenMP, independent per-thread
  problems contending only for shared bandwidth) and confirms the reversal the
  roofline term predicts: at compute intensity fops≈4–16 the verdict flips from
  MATERIALIZE (single-thread, compute-bound) to RECOMPUTE (all-core,
  bandwidth-bound). LESSON: the working set must exceed the shared LLC — the
  7950X3D's 128 MiB (2× 3D V-cache) L3 hid the reversal at small array sizes
  (all buffers cached → MATERIALIZE always won); only at ≥256 MiB aggregate does
  the buffer reload miss to DRAM and the bandwidth reversal appear. This is the
  decision the thread-blind model gets wrong; the `streamCycles` term gets it
  right per thread-count.
- ✅ **P2** — the III.4a workload-model fork is RESOLVED as **dual mode**
  (`costmodel_p2dual`): `thread.exclusive` selects shared-BW (interspersed) vs
  owned-BW (exclusive) in `streamCycles`; empirically justified (the modes have
  opposite optimal at fops≈4–16). The gap-#2 *capacity* double-count is handled
  by the same pairing — exclusive mode pairs with `llc_sharers=1` (the
  parallelized loop's `WS/N` vs `L3/N` cancels to `WS` vs full `L3`, no derate),
  interspersed with `llc_sharers=N`. `avx512FreqThrottle` resolves to a
  descriptive field that informs `vectorBitsNative` (a heavily-throttled all-core
  Xeon sets it to 256), not a separate runtime knob — just the Idun measurement
  that sets the value. Remaining P2 nicety: a per-candidate `isParallelLoop`
  auto-detect (`affine::isLoopParallel`) so the mode need not be machine-global —
  optional.
- **P1.5** ✅ (`costmodel_p1_5`) — symmetric roofline: recompute's source
  re-reads are bandwidth-priced too, so the model gains the compute-vs-bandwidth
  crossover the bench measures (instead of always favoring recompute under a
  thread JSON). The always-on `sourceThrashes` capacity clause stays as its
  complement.
- **P3** — SMT L1/L2 split ✅ (`costmodel_p3smt`: `effectivePrivateCache`,
  fission + DR L2 gates, makes `effectiveCache` live). **False-sharing padding is
  BLOCKED on missing infrastructure**: drcompiler emits SERIAL code (the cost
  model reasons about parallel *deployment* to pick the best serial code, but no
  pass emits OpenMP/pthreads), so there is no multi-thread write to the fission
  buffer to false-share yet. Wire it WITH a parallel-emission backend, not before.
- **P4** — NUMA (remote tier in `streamCycles`/`effectiveCache`): **unvalidatable
  on this single-socket host** and large; deferred until a multi-socket target
  (the model shape is a per-node BW + a remote-latency multiplier on
  `streamCycles`/the tier).
- **`isParallelLoop` auto-detect** (`affine::isLoopParallel` per candidate):
  low-value now that the workload mode is a machine-global JSON property
  (deployment is whole-program, not per-loop); revisit only if a single binary
  must mix exclusive and interspersed regions.

**Validation** (mirror `CONTENTION_AWARE_COSTMODEL.md`): a parallel
PolyBench/GEMM kernel swept over `activeThreads` ∈ {1, cores/2, cores}, on the
7950X3D (16 cores, 2 CCX → `coresPerLLC=8`) and on Idun. Pin the decision-vs-
measured table as a lit test; the headline target is a **BW-driven keep→recompute
reversal** (P1) and a **per-thread-WS tiling change** (P2) that the single-thread
model gets wrong.

**Risks / honest caveats:**
- Bandwidth numbers are the least-portable inputs; default them to 0 (disabled)
  so a machine without a measured BW behaves exactly as today.
- The roofline is a *first-order* model (no prefetcher, no MLP, no bank
  conflicts); it will misprice some kernels — but "has a BW axis at all" strictly
  dominates "latency+capacity only" for the parallel case.
- `coresPerLLC`/`smtPerCore`/`activeThreads` are operator-supplied (like
  `llcSharers` today) — the model predicts *given* the topology; it does not
  discover it.
- Keep the gate strict: this is a large surface, and the entire value of the v4
  unification is that `nThreads=1` stays bit-for-bit what it is now.
