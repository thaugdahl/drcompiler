# DataRecomputation Pass — Architecture

This document describes the post-refactor layout of the
`data-recomputation` pass. For the catalogue of rewrite strategies
themselves, see `REMAT_STRATEGIES.md`. For the migration plan that
produced this layout, see `STRATEGY_REFACTOR_PLAN.md`.

## File layout

```
include/drcompiler/Transforms/DataRecomputation/
    AnalysisState.h          # vocabulary types (StoreMap, LoadProvenanceMap, …)
    DotEmitter.h             # GraphViz dump of the provenance graph
    CacheCostModel.h         # cache params, footprint analysis, decideBufferStrategy
    RematKernel.h            # isRematerializable, rematerializeAt, partial-remat helpers
    Strategies/
        Strategy.h           # LoadStrategy interface + LoadCandidate / StrategyEnv / Outcome
        ConstantGlobalFold.h # S0
        DirectForward.h      # S1
        FullRemat.h          # S2
        PartialRemat.h       # S2b (Mode enum: Intra / Interproc)
        InterprocRemat.h     # S3
        CrossFnOrdered.h     # S4 family (straight-line, D, F.1, F.2)

lib/Transforms/                          # parallel layout
    DataRecomputation.cpp                # pass class + pipeline driver (~2160 LOC)
    DataRecomputation/
        CacheCostModel.cpp               (~600 LOC)
        RematKernel.cpp                  (~735 LOC)
        Strategies/
            ConstantGlobalFold.cpp       (~70 LOC)
            DirectForward.cpp            (~32 LOC)
            FullRemat.cpp                (~37 LOC)
            PartialRemat.cpp             (~104 LOC)
            InterprocRemat.cpp           (~67 LOC)
            CrossFnOrdered.cpp           (~876 LOC)
```

All translation units link into the existing `DRCompTransforms` MLIR
library — no new CMake target.

## Namespaces

| Namespace            | Contents                                              |
|----------------------|-------------------------------------------------------|
| `dr`                 | Analysis vocabulary, cost model, rematerialization kernel |
| `dr::strategies`     | `LoadStrategy` interface + per-strategy classes + S4 plan/materialize/reader-rewrite |
| `drcompiler`         | Pre-existing utility namespace (Utils/, CpuCostModel) |
| `mlir`               | Pass class, registration boilerplate                  |

## Pipeline flow

```text
DataRecomputationPass::runOnOperation()
  │
  ├── runConstantGlobalFold (S0 peephole, pre-analysis)
  │
  ├── allocation roots, store-value deps, module-wide writers
  │
  ├── per-function dataflow: analyzeBlock / analyzeOp
  │     → fills LoadProvenanceMap + InterproceduralOriginMap
  │
  ├── per-buffer cost decision: decideBufferStrategy
  │     → populates skipBuffers
  │
  ├── per-load strategy pipeline (first-match-wins):
  │     for each SINGLE candidate:
  │       skip if buffer in skipBuffers
  │       same-function?
  │         DirectForward  → FullRemat  → PartialRemat(Intra)
  │       cross-function?
  │         InterprocRemat (which itself falls back to PartialRemat(Interproc))
  │
  └── Strategy 4 — cross-fn ordered recomputation:
        group surviving SINGLE loads by (readerFn, storeOp)
        for each group:
          buildCrossFunctionRematPlan       (dispatches D / F.1 / F.2)
          findEligibleWriterCall per caller
          materializeAtCaller / Loop / Extraction
          addParamAndReplaceLoads + rewriteCallSite (in-place or specialized)
```

## Strategy interface

```cpp
namespace dr::strategies {

struct LoadCandidate {
  mlir::Operation *loadOp;
  mlir::Operation *storeOp;     // unique reaching store
  mlir::Value      storedValue;
};

struct StrategyEnv {            // wired once in runOnOperation
  mlir::DominanceInfo &dom;
  LoadProvenanceMap &loadProv;
  InterproceduralOriginMap &interprocOrigins;
  AllocationRoots &allocRootFor;
  const RootWriteMap &rootWrites;
  const CacheParams &cache;
  const drcompiler::CpuCostModel &costModel;
  bool partialRematEnabled;
  unsigned partialMaxLeaves;
  bool emitDiagnostics;
};

enum class Outcome { Accepted, NotApplicable };

class LoadStrategy {
public:
  virtual llvm::StringRef name() const = 0;
  virtual Outcome tryApply(LoadCandidate &c, StrategyEnv &env) = 0;
};

} // namespace dr::strategies
```

`Accepted` means the strategy consumed the candidate (rewrote the load
or rejected it after its own gate); the pipeline stops for that
candidate. `NotApplicable` means the strategy didn't fire and the next
one in the pipeline gets a turn.

S4 (`CrossFnOrdered`) does **not** implement `LoadStrategy` — it
operates on `(readerFn, storeOp)` groups, not single loads, and is
invoked as a separate phase after the per-load pipeline has run.

## Cost model surface

`CacheCostModel.h` exposes the cache-aware cost helpers:

- `CacheParams` — L1/L2/L3 size + latency, cache line size.
- `estimateBufferSizeBytes`, `estimateLoadLatency` — buffer-tier latency.
- `estimateInterveningFootprint` — bytes of memory traffic between a
  store and a load in the same function (handles nested scopes, loops,
  branches).
- `estimateOperandReloadPenalty` — cycles charged for cold operands
  needed by recomputation.
- `decideBufferStrategy` — combines the above into a per-buffer
  RECOMPUTE-vs-KEEP decision.
- `estimateTripCount`, `traceToConstant` — loop trip count helpers used
  by footprint estimation.

`RematKernel.h` exposes the IR-mutation primitives:

- `isRematerializable` — operand-tree walker, decides whether the SSA
  chain rooted at a stored value can be cloned at an insertion point.
  Supports load-chaining (via `LoadProvenanceMap`) and partial
  rematerialization (via `PartialRematOpts`).
- `rematerializeAt` — performs the cloning.
- `RootWriteMap` / `buildRootWriteMap` — module-wide per-allocation-root
  write summary used by partial-remat leaf safety.
- `estimateAccessLatency` / `estimateLeafLoadsCost` — stride-aware leaf
  cost for the partial-remat gate.

## What this refactor improved

1. **`runOnOperation` shrank from ~1245 lines of nested `if`/`else` to
   a ~50-line driver.** The per-load strategy pipeline reads top-to-
   bottom: build env, instantiate strategies, loop over candidates,
   pick the first one that fires.

2. **One concept per file.** Each strategy lives in its own header +
   .cpp pair. `REMAT_STRATEGIES.md` headings line up 1:1 with file
   names.

3. **Diagnostic strings co-located with the code that decides them.**
   Previously scattered across the giant `runOnOperation`; now the
   `direct-forward: ACCEPT/REJECT_*` remarks are emitted from
   `DirectForward.cpp` and so on.

4. **The cache cost model is reusable.** Other passes that want
   footprint-aware decisions (e.g. `memory-fission`) can include
   `CacheCostModel.h` instead of copy-pasting helpers.

5. **`isRematerializable` is the single source of truth for "can this
   SSA tree be cloned safely?"** All four rematerialization-bearing
   strategies (`FullRemat`, `PartialRemat`, `InterprocRemat`,
   `CrossFnOrdered`'s plan-builder) call it. They cannot drift on
   purity / dominance / chaining checks.

6. **S4's plan / materialize / reader-rewrite are now callable from
   tests.** Each of `buildCrossFunctionRematPlan`,
   `materializeExtraction`, `cloneReaderForSpec`, etc. is a public
   function in `dr::strategies`. Unit tests can construct an IR
   fragment and exercise just one of them.

## Extensibility

### Adding a new per-load strategy

1. Create `Strategies/MyStrategy.{h,cpp}` declaring a class deriving
   from `LoadStrategy`.
2. Implement `tryApply(LoadCandidate&, StrategyEnv&)`. Return
   `Accepted` after rewriting (or after a self-rejection like
   `REJECT_TYPE` that should stop the pipeline). Return
   `NotApplicable` to let later strategies have a go.
3. Add the `.cpp` to `lib/Transforms/CMakeLists.txt`.
4. Include the header in `DataRecomputation.cpp` and slot the
   instance into the pipeline at the desired priority. Exact order
   matters: today `DirectForward → FullRemat → PartialRemat(Intra)`
   intra and `InterprocRemat` cross-fn.

### Tuning the cost model

`CacheParams` is plain data, populated from CLI flags and optionally
overridden by a JSON file (`drcompiler::CpuCostModel`). To change a
gate, edit `decideBufferStrategy` (cache-fit gate) or
`estimateLeafLoadsCost` (partial-remat leaf gate). The strategy
classes consume the result; no strategy needs to be edited.

### Reordering or disabling strategies

The pipeline is built inline in `runOnOperation`. Today the order is
fixed; conditional construction (e.g. an
`--dr-disable-strategy=full-remat` flag) is a one-line change at the
pipeline build site — no surgery on individual strategies.

### Sharing analysis output across strategies

`StrategyEnv` is the contract between the driver and any strategy.
Add a field there to thread new per-pass state into all strategies at
once.

## Future design directions

These are not in scope for the current refactor but are unblocked by
it.

### 1. Strategy registration via a registry

Today the pipeline is hand-built in `runOnOperation`. A
`LoadStrategyRegistry` (string name → factory) would let:

- the test harness assert per-strategy behaviour by spinning up only
  one strategy in isolation;
- a `--dr-strategies=direct-forward,full-remat` flag select the
  pipeline at runtime;
- third-party out-of-tree consumers register their own strategies
  without forking the pass.

### 2. Per-strategy verbosity / metrics

Each strategy already knows its `name()`. Wiring that into
`PassInstrumentation` would give `mlir-opt --pass-statistics` style
output: how many candidates each strategy fires on, per-strategy
reject-reason histogram. Currently this is only obtainable by parsing
`-dr-test-diagnostics` remarks.

### 3. Split `CrossFnOrdered.cpp`

At ~876 LOC it is the largest single file. Natural fault lines (per
`STRATEGY_REFACTOR_PLAN.md`):

```
CrossFnOrdered/
    PlanBuilder.cpp          # buildCrossFunctionRematPlan + buildStraightLinePlan / buildLoopPlan / walkStoredValueTree
    CallerSelection.cpp      # findEligibleWriterCall
    Materialize.cpp          # materializeAtCaller / materializeLoop / materializeExtraction
    ReaderRewrite.cpp        # addParamAndReplaceLoads / addMemrefParamAndRewriteLoads / cloneReaderForSpec / rewriteCallSite
    CrossFnOrdered.cpp       # the group-driver glue (currently sits in runOnOperation)
```

Each piece is independently testable. The driver could become a
`GroupedStrategy` class mirroring `LoadStrategy`.

### 4. Lift the analysis kernels next

`runOnOperation` still owns the analysis-engine glue (~600 lines):
allocation-root collection, store-value deps, module-wide writer
tables, phase-root computation, the `analyzeBlock` driver. The next
refactor should extract these to `Analysis/StoreProvenance.{h,cpp}` +
`Analysis/CallGraph.{h,cpp}`, mirroring the strategy split. This
would shrink `DataRecomputation.cpp` from ~2160 LOC to ~500–700 LOC
of pass plumbing.

### 5. Strategy 2b cost-gate unification

`PartialRemat` currently has two cost gates selected by `Mode`:

- `Intraprocedural` — strict `alu + leaf < loadLat` check.
- `Interprocedural` — full `decideBufferStrategy` with
  `numConsumers=1`, `storeToLoadFootprint=0`, `operandPenalty=0`.

These should converge on `decideBufferStrategy` in both modes, with
the right footprint estimate plumbed in. The split is a latent
asymmetry, made visible by the extraction.

### 6. Loop-carried state in S4 (Strategy F)

`buildLoopPlan` rejects loops with `iter_args` (no loop-carried
state). Lifting this is the next big win for Marco workloads where
reduction loops dominate.

### 7. Strategy 4 cost gate refinement

S4 currently inherits the per-load cost gate (`skipBuffers`). A
proper cost model for S4 would compare:

- the materialization-at-caller cost (clone the writer's expression
  once per ordered call), versus
- the saved-load cost (one fewer memory access per reader execution).

These differ from the per-load gate because the materialization is
amortized across a whole reader call.
