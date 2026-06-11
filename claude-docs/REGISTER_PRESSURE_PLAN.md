# Register Pressure Extension Plan

**Goal:** Extend drcompiler's analytical cost model to jointly reason about
cache hierarchy *and* register pressure, then validate by replacing the
placeholder cost model in MLIR's `--affine-loop-fusion` and evaluating on
PolyBench.

**Estimated effort:** 4–8 weeks for full pipeline. Each phase is independently
useful — incremental value if scope shrinks.

---

## 0. Decisions locked in (from clarification round)

| Question | Choice |
|----------|--------|
| Target architecture | Configurable via `costs.json`; per-arch handlers with `generic` fallback; triplet-based dispatch |
| Scope | Full pipeline (analysis → DR integration → fusion fork → PolyBench) |
| Cost combination | Weighted sum: `total = α·memCycles + β·regCycles + γ·aluCycles`, weights per-arch |
| Fusion integration | Fork upstream `--affine-loop-fusion` into drcompiler |
| Vector register model | Hybrid: per-arch handler decides; default 1:1 |
| Spill aggregation | Strategy-driven; swap between excess-at-hot-point, sum-of-excess, graph-coloring |
| Code location | New `lib/Analysis/` directory |
| API surface | MLIR `AnalysisManager`-managed (cached, invalidation-aware) |

---

## 1. Architecture overview

```
                       ┌─────────────────────────────┐
                       │  UnifiedCostModel (facade)  │
                       └──────────────┬──────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
   ┌──────────▼─────────┐  ┌──────────▼─────────┐  ┌──────────▼─────────┐
   │  CacheCostModel    │  │ RegisterPressure   │  │   CpuCostModel     │
   │  (existing)        │  │ Analysis           │  │   (existing)       │
   └──────────┬─────────┘  └──────────┬─────────┘  └──────────┬─────────┘
              │                       │                       │
              └───────────────────────┼───────────────────────┘
                                      │
                          ┌───────────▼───────────┐
                          │   ArchHandler         │
                          │   (generic | x86-64-  │
                          │    avx2 | avx512 |    │
                          │    arm-neon | ...)    │
                          └───────────┬───────────┘
                                      │
                          ┌───────────▼───────────┐
                          │   costs.json + arch   │
                          │   block (triplet,     │
                          │   handler, params)    │
                          └───────────────────────┘
```

The **`UnifiedCostModel`** is the facade callers use. It delegates per-aspect
cost queries to the three component models, then combines results using the
arch handler's weights. The **`ArchHandler`** abstracts target-specific
decisions: register counts, vector width legalization, spill cost calibration.

---

## 2. Configuration extension (costs.json)

### 2.1 Current format (preserve)

```json
{
  "default_cost": 5,
  "ops": { "arith.addi": 1, "math.sqrt": 20, ... },
  "cache": {
    "l1_size": 32768,
    "l2_size": 262144,
    "l3_size": 33554432,
    "l1_latency": 4,
    "l2_latency": 12,
    "l3_latency": 40,
    "mem_latency": 200
  }
}
```

### 2.2 New extensions

```json
{
  // ... existing fields ...
  "arch": {
    "triplet": "x86_64-unknown-linux-gnu",
    "handler": "x86-64-avx2",          // or "generic", "x86-64-avx512", "arm-neon"
    "vector_width_bits": 256,           // physical vector reg width
    "spill_strategy": "excess-hot",     // or "sum-excess", "graph-color"
    "weights": {
      "alpha_mem": 1.0,                 // memory cycle weight
      "beta_reg": 1.0,                  // register spill cycle weight
      "gamma_alu": 1.0                  // ALU cycle weight
    }
  },
  "registers": {
    "gp_budget": 16,
    "vec_budget": 16,                   // 32 for AVX-512
    "pred_budget": 0,                   // 8 for AVX-512
    "spill_reload_cycles": 5,
    "spill_store_cycles": 1
  }
}
```

### 2.3 Auto-probing extension (scripts/gen_cpu_cost_model.py)

Add register inference:
- Parse `/proc/cpuinfo` for AVX2/AVX-512 flags → set vector budget
- Parse `lscpu` for architecture → set triplet
- Use `llvm-mca` to estimate realistic spill cycles by compiling a known
  high-pressure kernel and measuring resource pressure
- Tune weights `(α, β, γ)` empirically by running a calibration suite

---

## 3. Architecture handler system

### 3.1 Interface

```cpp
// include/drcompiler/Analysis/ArchHandler.h

namespace drcompiler {

struct ArchParams {
  llvm::Triple triple;
  unsigned vectorWidthBits = 128;
  // weights for unified cost combiner
  double alphaMem = 1.0, betaReg = 1.0, gammaAlu = 1.0;
};

class ArchHandler {
public:
  virtual ~ArchHandler() = default;

  /// Map an MLIR type to register class + count.
  virtual RegClassRequirement classifyType(mlir::Type) const = 0;

  /// Adjust spill cost estimate per architecture (e.g., AVX-512 has
  /// different spill characteristics from SSE).
  virtual unsigned tuneSpillCost(unsigned base, RegClass) const = 0;

  /// Arch-specific veto: e.g., AVX-512 mask spills are expensive.
  virtual std::optional<std::string> checkConstraint(
      const PressureResult &, const RegisterParams &) const = 0;

  /// Combine the three cost components per architecture.
  virtual unsigned combineCosts(unsigned memCycles,
                                unsigned regCycles,
                                unsigned aluCycles,
                                const ArchParams &) const;

  static std::unique_ptr<ArchHandler> create(StringRef handlerName);
};

enum class RegClass { GP, Vec, Pred };

struct RegClassRequirement {
  RegClass cls;
  unsigned count = 1;   // e.g., vector<16xf32> on AVX2 = 2 regs
};

}  // namespace drcompiler
```

### 3.2 Concrete handlers

```cpp
// lib/Analysis/ArchHandlers/Generic.cpp        — 1:1, no special veto
// lib/Analysis/ArchHandlers/X86_64_AVX2.cpp    — 256-bit vec, no pred regs
// lib/Analysis/ArchHandlers/X86_64_AVX512.cpp  — 512-bit vec, 8 mask regs
// lib/Analysis/ArchHandlers/ARM_Neon.cpp       — 128-bit vec, 32 vec regs
```

Each handler is ~100 LOC. Register a `StringMap<HandlerFactory>` for dispatch.

### 3.3 Dispatch logic

```cpp
// At cost-model file load:
auto handlerName = jsonRoot["arch"]["handler"].getAsString().value_or("generic");
auto handler = ArchHandler::create(handlerName);
if (!handler) {
  emitWarning("unknown arch handler '" + handlerName + "', falling back to generic");
  handler = ArchHandler::create("generic");
}
```

---

## 4. RegisterPressureAnalysis

### 4.1 Public API

```cpp
// include/drcompiler/Analysis/RegisterPressureAnalysis.h

namespace drcompiler {

struct PressureResult {
  // Per-class peak live values across the region.
  llvm::DenseMap<RegClass, unsigned> peakLive;
  // Per-class excess (live > budget) at each program point, summarized.
  llvm::DenseMap<RegClass, unsigned> totalExcess;
  // Total estimated spill reload cycles (already aggregated by strategy).
  uint64_t totalSpillCycles = 0;
  // Per-block / per-op live counts (cached, queried lazily).
  llvm::DenseMap<mlir::Operation *, unsigned> liveAtOp;
};

/// AnalysisManager-managed analysis. Cached per Operation, invalidated when
/// the operation is mutated.
class RegisterPressureAnalysis {
public:
  RegisterPressureAnalysis(mlir::Operation *op,
                           const RegisterParams &params,
                           const ArchHandler &arch,
                           SpillStrategy strategy = SpillStrategy::ExcessHot);

  /// Pressure summary for a region (e.g., a loop body or function).
  PressureResult analyzeRegion(mlir::Region &region);

  /// Pressure if `region` were transformed by adding `clonedOps` (used for
  /// remat / fusion hypothetical queries).
  PressureResult analyzeHypothetical(mlir::Region &region,
                                     llvm::ArrayRef<mlir::Operation *> clonedOps);

  /// Free function form for one-shot queries.
  static PressureResult analyzeRegionStatic(mlir::Region &region,
                                            const RegisterParams &params,
                                            const ArchHandler &arch,
                                            SpillStrategy strategy);

private:
  // ... liveness cache, op-to-pressure map ...
};

enum class SpillStrategy {
  ExcessHot,     // max(0, peakLive - budget) * spillReload * tripCount
  SumExcess,     // sum over program points of max(0, live - budget)
  GraphColor     // interference-graph approximation
};

}  // namespace drcompiler
```

### 4.2 Implementation strategy

**Step 1: Liveness computation.** Reuse `mlir::Liveness`. Walk the region's
blocks; for each block, get `LivenessBlockInfo`. Iterate operations within the
block in order, tracking the set of values live at each program point.

**Step 2: Classification.** For each live value at each program point, call
`arch.classifyType(value.getType())` to get the register class and count.
Accumulate per-class live counts.

**Step 3: Peak/excess computation.** Track per-class peak across the region.
Compare against budget to compute excess.

**Step 4: Spill aggregation (strategy-driven).** Three implementations:

```cpp
// lib/Analysis/SpillStrategies/ExcessHotStrategy.cpp
uint64_t ExcessHotStrategy::aggregate(const LivenessTrace &trace,
                                       const RegisterParams &params,
                                       uint64_t tripCount) {
  uint64_t total = 0;
  for (auto [cls, peak] : trace.perClassPeak) {
    unsigned budget = params.budgetFor(cls);
    if (peak > budget) {
      unsigned excess = peak - budget;
      total += uint64_t(excess) * params.spillReloadCycles * tripCount;
    }
  }
  return total;
}

// lib/Analysis/SpillStrategies/SumExcessStrategy.cpp
// Walks each program point, sums excess.

// lib/Analysis/SpillStrategies/GraphColorStrategy.cpp
// Builds interference graph from liveness, runs Chaitin-style coloring,
// counts forced spills.
```

The strategy is selected via the cost-model JSON's `"spill_strategy"` field
(or via a pass option that overrides).

**Step 5: Hypothetical analysis (for remat queries).** When the cost model
asks "what if I cloned these ops at this load site?", build a hypothetical
liveness view: original live set ∪ (def-use closure of cloned ops). Reuse the
same aggregation strategy.

### 4.3 Caching and invalidation

Register as an `AnalysisManager`-managed analysis on `func::FuncOp` (the
natural granularity for register pressure). MLIR will invalidate when the
function is mutated. Within a single pass, multiple consultations are O(1)
after the first.

```cpp
// In a pass:
auto &rpa = getAnalysis<RegisterPressureAnalysis>();
auto result = rpa.analyzeRegion(loopBody);
```

---

## 5. UnifiedCostModel facade

### 5.1 Interface

```cpp
// include/drcompiler/Analysis/UnifiedCostModel.h

namespace drcompiler {

class UnifiedCostModel {
public:
  UnifiedCostModel(const CacheParams &cache,
                   const RegisterParams &regs,
                   const CpuCostModel &cpu,
                   const ArchParams &arch,
                   std::unique_ptr<ArchHandler> handler);

  /// Per-buffer rematerialization decision (extends decideBufferStrategy).
  struct RematDecision {
    bool recompute;
    unsigned memCycles, regCycles, aluCycles, totalCycles;
    PressureResult predictedPressure;
    std::string rationale;   // human-readable for diagnostics
  };

  RematDecision decideRemat(/* inputs from existing decideBufferStrategy */
                             unsigned aluCost, unsigned leafLoadCost,
                             unsigned loadLatency, unsigned numConsumers,
                             int64_t bufferSizeBytes,
                             /* NEW: */
                             mlir::Region &consumerRegion,
                             llvm::ArrayRef<mlir::Operation *> rematOps);

  /// Affine loop fusion decision.
  struct FusionDecision {
    bool fuse;
    unsigned bestDepth;
    unsigned memCycles, regCycles, aluCycles, totalCycles;
    unsigned memCyclesUnfused, regCyclesUnfused, aluCyclesUnfused;
    std::string rationale;
  };

  FusionDecision decideFusion(mlir::affine::AffineForOp srcLoop,
                               mlir::affine::AffineForOp dstLoop,
                               llvm::ArrayRef<unsigned> candidateDepths);

  /// Loop tiling tile-size selection.
  struct TilingDecision {
    llvm::SmallVector<unsigned, 4> tileSizes;
    unsigned predictedSpills;
    int64_t predictedFootprint;
    std::string rationale;
  };

  TilingDecision decideTileSizes(llvm::ArrayRef<mlir::affine::AffineForOp> band);

  /// Loop unroll factor selection.
  unsigned chooseUnrollFactor(mlir::affine::AffineForOp loop,
                               llvm::ArrayRef<unsigned> candidateFactors);

private:
  // ... members ...
};

}  // namespace drcompiler
```

### 5.2 Combiner

```cpp
unsigned UnifiedCostModel::combine(unsigned mem, unsigned reg, unsigned alu) const {
  return handler->combineCosts(mem, reg, alu, archParams);
}

// Default combiner in ArchHandler::combineCosts:
unsigned ArchHandler::combineCosts(unsigned mem, unsigned reg, unsigned alu,
                                    const ArchParams &p) const {
  return unsigned(p.alphaMem * mem + p.betaReg * reg + p.gammaAlu * alu);
}
```

Per-arch handlers may override (e.g., AVX-512 mask spills get a multiplier).

---

## 6. DR integration (Phase 3)

### 6.1 Extend `decideBufferStrategy`

Current signature (in `CacheCostModel.h`):

```cpp
MaterializationDecision decideBufferStrategy(
  unsigned aluCost, unsigned leafLoadCost, unsigned loadLatency,
  unsigned numConsumers, int64_t bufferSizeBytes,
  int64_t storeToLoadFootprint, unsigned operandPenalty,
  const CacheParams &cache);
```

New signature:

```cpp
MaterializationDecision decideBufferStrategy(
  unsigned aluCost, unsigned leafLoadCost, unsigned loadLatency,
  unsigned numConsumers, int64_t bufferSizeBytes,
  int64_t storeToLoadFootprint, unsigned operandPenalty,
  const CacheParams &cache,
  /* NEW: */
  const RegisterParams &regs,
  const ArchHandler &arch,
  mlir::Region &consumerRegion,
  llvm::ArrayRef<mlir::Operation *> rematOps);
```

Decision becomes:

```cpp
unsigned memCycles_keep      = numConsumers * loadLatency;
unsigned memCycles_recompute = leafLoadCost;
unsigned aluCycles_keep      = aluCost + 1;            // compute once + store
unsigned aluCycles_recompute = numConsumers * aluCost;

// NEW: register cost
auto pressureKeep = rpa.analyzeRegion(consumerRegion);
auto pressureRecompute = rpa.analyzeHypothetical(consumerRegion, rematOps);

unsigned regCycles_keep      = pressureKeep.totalSpillCycles;
unsigned regCycles_recompute = pressureRecompute.totalSpillCycles;

unsigned totalKeep      = arch.combineCosts(memCycles_keep, regCycles_keep, aluCycles_keep, arch.params());
unsigned totalRecompute = arch.combineCosts(memCycles_recompute, regCycles_recompute, aluCycles_recompute, arch.params());

decision.recompute = totalRecompute < totalKeep;
decision.rationale = formatRationale(...);  // for diagnostics
```

### 6.2 Plumbing in DataRecomputation.cpp

Find all call sites of `decideBufferStrategy` (currently several in the pass).
At each, pass through:
- The consumer region (load's enclosing region)
- The hypothetical remat ops (the computation tree being cloned)
- The `RegisterParams` and `ArchHandler` from the pass options / cost-model JSON

Add new pass options:
```
dr-reg-budget-gp:    16
dr-reg-budget-vec:   16
dr-reg-budget-pred:   0
dr-spill-reload:      5
dr-spill-store:       1
dr-spill-strategy:    excess-hot | sum-excess | graph-color
dr-arch-handler:      generic | x86-64-avx2 | x86-64-avx512 | arm-neon
dr-cost-weight-mem:   1.0
dr-cost-weight-reg:   1.0
dr-cost-weight-alu:   1.0
```

JSON file overrides take precedence (already the pattern for cache params).

### 6.3 Diagnostics extension

Existing remarks (`cost-model: RECOMPUTE/KEEP`) get extended to include
register cost breakdown:

```
cost-model: KEEP (alu=8, mem=12, reg=0, total=20)  →
  vs RECOMPUTE (alu=32, mem=2, reg=15, total=49)
```

This makes the decision auditable in tests and debug runs.

---

## 7. Affine loop fusion fork (Phase 4)

### 7.1 Setup

Create `lib/Transforms/AffineLoopFusionCostModel/`:
- Copy `mlir/lib/Dialect/Affine/Transforms/LoopFusion.cpp` from upstream
  (the current MLIR 22 version)
- Add it as a new pass `dr-affine-loop-fusion` in `Passes.td`
- Track upstream version (record the commit SHA) for periodic re-sync

### 7.2 Replace `isFusionProfitable`

The placeholder at LoopFusion.cpp:657 is the surgical target. Replace:

```cpp
// OLD: placeholder
if ((storageReduction > maxStorageReduction) &&
    (additionalComputeFraction <= computeToleranceThreshold)) {
  maxStorageReduction = storageReduction;
  bestDstLoopDepth = i;
  ...
}
```

With:

```cpp
// NEW: unified cost model
auto decision = costModel->decideFusion(srcForOp, dstForOp, candidateDepths);
if (decision.fuse) {
  *dstLoopDepth = decision.bestDepth;
  return true;
}
return false;
```

### 7.3 `decideFusion` implementation

```cpp
FusionDecision UnifiedCostModel::decideFusion(
    affine::AffineForOp src, affine::AffineForOp dst,
    ArrayRef<unsigned> depths) {

  // Phase 1: Unfused baseline cost
  auto baselineSrc = analyzeLoop(src);
  auto baselineDst = analyzeLoop(dst);
  unsigned memUnfused = baselineSrc.memCycles + baselineDst.memCycles
                        + intermediateBufferAccessCycles(src, dst);
  unsigned regUnfused = baselineSrc.regCycles + baselineDst.regCycles;
  unsigned aluUnfused = baselineSrc.aluCycles + baselineDst.aluCycles;
  unsigned totalUnfused = combine(memUnfused, regUnfused, aluUnfused);

  // Phase 2: For each candidate fusion depth, estimate fused cost
  FusionDecision best;
  best.totalCycles = totalUnfused;
  best.fuse = false;

  for (unsigned depth : depths) {
    auto sliced = computeSliceBounds(src, dst, depth);
    if (!sliced) continue;

    unsigned memFused = estimateFusedMemCost(src, dst, *sliced);
    auto pressure = rpa.analyzeHypotheticalFusion(src, dst, *sliced);
    unsigned regFused = pressure.totalSpillCycles;
    unsigned aluFused = baselineSrc.aluCycles + baselineDst.aluCycles
                         + sliced->additionalCompute;
    unsigned totalFused = combine(memFused, regFused, aluFused);

    if (totalFused < best.totalCycles) {
      best.totalCycles = totalFused;
      best.bestDepth = depth;
      best.fuse = true;
      best.memCycles = memFused;
      best.regCycles = regFused;
      best.aluCycles = aluFused;
      best.rationale = ...;
    }
  }

  best.memCyclesUnfused = memUnfused;
  best.regCyclesUnfused = regUnfused;
  best.aluCyclesUnfused = aluUnfused;
  return best;
}
```

The novel piece is `pressure.analyzeHypotheticalFusion` — given two loops and
a slice depth, predict the live set of the fused body. Implementation: clone
the source slice into the destination at the requested depth, run liveness
on the synthetic combined region, then discard.

### 7.4 Upstream-tracking discipline

- Pin to a known upstream commit. Document in `lib/Transforms/AffineLoopFusionCostModel/UPSTREAM.md`.
- Periodically diff against upstream, cherry-pick correctness fixes.
- When publishing, the diff against upstream IS the contribution — keep it
  minimal and surgical to make review easy.

---

## 8. PolyBench evaluation (Phase 5)

### 8.1 Methodology

For each PolyBench kernel:

1. Compile via cgeist → affine MLIR (already exists in `scripts/polybench-bench.sh`).
2. Run with 4 fusion configurations:
   - **none:** no fusion pass
   - **upstream:** stock `--affine-loop-fusion` (placeholder cost model)
   - **drcomp:** `--dr-affine-loop-fusion` with unified cost model
   - **drcomp-maxfuse:** unified cost model with `α_mem` boosted (aggressive fusion baseline for comparison)
3. Lower → mlir-translate → clang -O2 → time (median of N≥7 runs).
4. Collect for each kernel: `time`, `predicted_spills`, `actual_spills`
   (from `llvm-mca`), `decision_rationale`.

### 8.2 Hypothesis tests

- **H1: drcomp avoids regressions.** Count kernels where `upstream < none` (i.e., upstream over-fused). Verify drcomp matches or beats `none` on those.
- **H2: drcomp preserves wins.** Count kernels where `upstream > none` (legitimate wins). Verify drcomp matches upstream on those.
- **H3: predicted spills correlate with measured spills.** Pearson correlation between drcomp's predicted spill count and llvm-mca's reported spill count. Target r > 0.7.
- **H4: weighted-sum policy beats hard-constraint baseline.** Compare against an ablation where regCost > 0 → reject fusion outright.

### 8.3 Calibration suite

Before running PolyBench, calibrate `(α, β, γ)` on a held-out calibration set
(e.g., 5 kernels not in PolyBench, or synthetic micro-kernels). Use grid
search or gradient-free optimization (Nelder-Mead via scipy). Report
calibrated weights and sensitivity analysis.

### 8.4 Reporting

CSV output per run, plus a summary table:

```
Kernel    | none | upstream | drcomp | upstream Δ | drcomp Δ | predicted spills | actual spills
----------+------+----------+--------+------------+----------+------------------+--------------
atax      | 60   | 81       | 60     | -35%       |   0%     | 14               | 16
2mm       | 1152 | 1159     | 1155   |  -1%       |  -0%     |  0               |  0
syr2k     | 1392 | 1378     | 1380   |  +1%       |  +1%     |  0               |  0
...
```

---

## 9. Implementation phases (sequenced)

### Phase 1: Architecture handler infrastructure (week 1)

**Deliverables:**
- `include/drcompiler/Analysis/ArchHandler.h`
- `lib/Analysis/ArchHandlers/{Generic, X86_64_AVX2, X86_64_AVX512, ARM_Neon}.cpp`
- `lib/Analysis/ArchHandlerRegistry.cpp` with `StringMap` dispatch
- Extension of `CpuCostModel::loadFromFile` to parse `"arch"` and `"registers"` blocks
- Unit tests: feed each `costs.json` variant, verify correct handler chosen and params loaded

**Files touched:** 8 new, 1 modified (`CpuCostModel.cpp`)

**Exit criteria:** `dr-opt --print-arch-handler` shows correct handler for each test JSON.

### Phase 2: RegisterPressureAnalysis (weeks 2–3)

**Deliverables:**
- `include/drcompiler/Analysis/RegisterPressureAnalysis.h`
- `lib/Analysis/RegisterPressureAnalysis.cpp`
- `lib/Analysis/SpillStrategies/{ExcessHot, SumExcess, GraphColor}.cpp`
- AnalysisManager registration in pass infrastructure
- Synthetic test suite: 20 MLIR programs with known live counts; verify analysis returns correct peak

**Files touched:** 6 new

**Validation:**
- Write 20 synthetic programs with deliberately constructed live sets (e.g., a loop body using 8 f64 values simultaneously, expect peakLive[Vec]=8 on x86-64-avx2 with 256-bit width).
- Verify each strategy produces consistent answers on simple cases; document differences on complex cases.

**Exit criteria:** Pearson correlation > 0.85 between `RegisterPressureAnalysis.totalSpillCycles` and `llvm-mca` spill cycle counts on a 20-program calibration set.

### Phase 3: DR integration (week 4)

**Deliverables:**
- Extended `decideBufferStrategy`, `decideBufferElimination` signatures
- Updated call sites in `DataRecomputation.cpp`
- Updated diagnostic remarks
- New pass options for register params
- 30 new synthetic tests in `bench/H/` (register-pressure-sensitive scenarios)

**Files touched:** 4 modified (CacheCostModel.h, CacheCostModel.cpp, DataRecomputation.cpp, Passes.td)

**Exit criteria:** All existing tests still pass. New tests demonstrate register pressure affects DR decisions (e.g., a load that DR previously recomputed now gets KEEP because remat would exceed register budget).

### Phase 4: Affine loop fusion fork (weeks 5–6)

**Deliverables:**
- `lib/Transforms/AffineLoopFusionCostModel/LoopFusion.cpp` (forked)
- `lib/Transforms/AffineLoopFusionCostModel/UPSTREAM.md` (tracking)
- New pass `dr-affine-loop-fusion` registered in `Passes.td` and `dr-opt`
- 15 lit tests reproducing PolyBench-like patterns at small scale

**Files touched:** 3 new, 2 modified (Passes.td, dr-opt.cpp)

**Exit criteria:** `dr-affine-loop-fusion` passes all upstream `--affine-loop-fusion` lit tests (correctness preserved) AND makes different decisions on the new register-pressure-sensitive tests.

### Phase 5: PolyBench evaluation (weeks 7–8)

**Deliverables:**
- Extended `scripts/polybench-bench.sh` with `drcomp` and `drcomp-maxfuse` configurations
- Calibration script `scripts/calibrate_weights.py`
- Results processing script generating the summary table
- 4 figures: time-per-kernel bar chart, predicted-vs-actual spill scatter, weight sensitivity heatmap, decision-flip-explanation table

**Files touched:** 3 modified, 2 new

**Exit criteria:** All 30 PolyBench kernels timed; calibrated weights produce ≥1 kernel where drcomp beats upstream by ≥1.10x, AND drcomp has zero kernels worse than `none` by more than the noise floor.

---

## 10. Validation strategy

### 10.1 Layered validation

**Layer 1 — Analysis correctness:** Hand-crafted MLIR programs with known
peak live counts. Verify the analysis matches.

**Layer 2 — Spill prediction accuracy:** Compile the same program through
clang -O2, extract spill count from LLVM `-mllvm -print-after=regallocfast`
or `llvm-mca` resource pressure. Verify `predicted ≈ actual ± 20%`.

**Layer 3 — Decision quality:** Run DR on the existing 202-test benchmark
suite. Verify the augmented cost model doesn't regress any existing wins.
Investigate every changed decision.

**Layer 4 — Runtime:** PolyBench end-to-end. The headline experiment.

### 10.2 Failure modes to watch for

- **Over-prediction of spills.** If `predicted_spills >> actual_spills`,
  the cost model will be too conservative (reject fusions that would actually
  be fine). Symptom: drcomp matches `none` on too many kernels.
- **Under-prediction of spills.** If `predicted < actual`, drcomp accepts
  bad fusions. Symptom: drcomp matches upstream's regressions.
- **Weight calibration overfits the calibration set.** Symptom: drcomp wins
  on calibration but loses on held-out PolyBench. Mitigation: keep
  calibration set strictly disjoint from evaluation.
- **Strategy choice matters too much.** If `excess-hot` and `sum-excess`
  give very different runtime predictions, the model is fragile. Mitigation:
  default to `excess-hot`; report ablation comparing all three.

---

## 11. Risk register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| LLVM regalloc differs from prediction more than expected | Medium | High | Calibration phase; if r < 0.5, fall back to coarser heuristic |
| atax regression doesn't reproduce broadly across PolyBench | Medium | High | Phase 5 first half is exploratory; if only 1-2 kernels show regressions, sharpen the thesis to "we identify specific PolyBench kernels where the placeholder fails" rather than "broad speedup" |
| Bondhugula has work-in-progress on this | Low | Medium | Check Discourse and GitHub PRs again at Phase 4 start; coordinate if needed |
| Upstream fusion correctness fixes (issues #61604 etc.) break our fork | Medium | Low | Track upstream commits; cherry-pick fixes monthly |
| Register pressure at MLIR level is too imprecise (LLVM does too much) | Medium | High | Phase 2 exit criterion (r > 0.85) is the canary; if we can't hit it, the whole approach is suspect — pivot the thesis |
| Weighted-sum combiner has too many tunables to defend | Low | Medium | Report sensitivity analysis; show results across (α, β, γ) ∈ {0.5, 1.0, 2.0}³ — if winners are stable, weights are defensible |
| PolyBench is too small/regular to show benefit | Medium | Medium | Add ONNX-MLIR output to evaluation (the 204-buffer / 751-loop ResNet50) — different program structure may reveal different opportunities |

---

## 12. Concrete file layout

```
onnx-mlir/
├── include/drcompiler/Analysis/
│   ├── ArchHandler.h                       [NEW]
│   ├── RegisterPressureAnalysis.h          [NEW]
│   ├── UnifiedCostModel.h                  [NEW]
│   └── SpillStrategy.h                     [NEW]
├── lib/Analysis/
│   ├── CMakeLists.txt                      [NEW]
│   ├── ArchHandler.cpp                     [NEW]
│   ├── ArchHandlers/
│   │   ├── Generic.cpp                     [NEW]
│   │   ├── X86_64_AVX2.cpp                 [NEW]
│   │   ├── X86_64_AVX512.cpp               [NEW]
│   │   └── ARM_Neon.cpp                    [NEW]
│   ├── RegisterPressureAnalysis.cpp        [NEW]
│   ├── SpillStrategies/
│   │   ├── ExcessHot.cpp                   [NEW]
│   │   ├── SumExcess.cpp                   [NEW]
│   │   └── GraphColor.cpp                  [NEW]
│   └── UnifiedCostModel.cpp                [NEW]
├── lib/Transforms/
│   ├── DataRecomputation.cpp               [MODIFIED — plumb register params]
│   ├── DataRecomputation/
│   │   ├── CacheCostModel.h                [MODIFIED — extended signatures]
│   │   └── CacheCostModel.cpp              [MODIFIED]
│   └── AffineLoopFusionCostModel/          [NEW DIR]
│       ├── LoopFusion.cpp                  [forked from upstream]
│       ├── LoopFusionUtils.cpp             [forked from upstream if needed]
│       └── UPSTREAM.md                     [NEW — tracks upstream SHA]
├── include/drcompiler/Transforms/Passes.td  [MODIFIED — new pass + options]
├── tools/dr-opt/dr-opt.cpp                  [MODIFIED — register new pass]
├── scripts/
│   ├── gen_cpu_cost_model.py                [MODIFIED — emit arch + registers blocks]
│   ├── polybench-bench.sh                   [MODIFIED — add drcomp + drcomp-maxfuse]
│   └── calibrate_weights.py                 [NEW]
└── bench/H/                                 [NEW — register-pressure synthetic tests]
    ├── H001-...  through H030-...           [NEW — 30 programs]
```

**LOC estimate:** ~3500 lines new code, ~500 lines modified.

---

## 13. Open questions / decision points (still ambiguous)

These weren't covered by the clarification round; I've stated my default
choice but flagging in case you want to override.

1. **Default `(α_mem, β_reg, γ_alu)` weights.**
   - My default: `(1.0, 1.0, 1.0)` — all cycles weighted equally as a starting
     point, then calibrate per-arch via `calibrate_weights.py`.
   - Alternative: per-arch defaults derived from microarchitecture analysis
     (e.g., AVX-512 spills cost more proportionally → β > 1).

2. **What counts as a "vector type" for the GP vs Vec classification?**
   - My default: `vector<...>` → Vec; scalar `f*` → Vec on x86-64 (since FP
     uses XMM); scalar `i*`/`index` → GP; `memref<...>` → 1 GP register
     (descriptor pointer).
   - Alternative: scalar `f*` → GP-FP class as a separate class. More
     accurate for some archs but adds complexity.

3. **Tile-size selection: LOCKED IN.** Include `--dr-affine-loop-tile` fork
   in this plan (middle path). Adds ~2 weeks (Phase 4.5). See §16 below for
   the tiling delta.

4. **Should the GraphColor strategy be available in v1, or stub it?**
   - My default: implement all three. GraphColor is ~300 LOC; useful for
     ablation studies in the paper.
   - Alternative: stub GraphColor (return ExcessHot result), implement
     properly only if calibration shows the other two are inadequate.

5. **Cross-region pressure (e.g., function body live values that survive a
     call) — model or ignore?**
   - My default: ignore (intra-region pressure only). Functions are
     analyzed independently.
   - Alternative: cross-call liveness via interprocedural analysis. Matches
     drcompiler's existing interprocedural infrastructure but adds complexity.

6. **AnalysisManager scope: `func::FuncOp` or `ModuleOp`?**
   - My default: `func::FuncOp` — natural granularity, register pressure
     doesn't typically cross function boundaries (calls are clobber points).
   - Alternative: `ModuleOp` — would enable cross-function pressure
     reasoning if (5) is also chosen.

---

## 14. Concrete first commit

To validate the plan minimally:

1. Create `lib/Analysis/` directory with empty `CMakeLists.txt`.
2. Stub `ArchHandler::create("generic")` returning a default-construct
   handler that always says "GP class, 1 register, no special veto."
3. Stub `RegisterPressureAnalysis::analyzeRegion` returning a hardcoded
   `PressureResult` with `peakLive = {GP: 0, Vec: 0}`.
4. Add `--print-register-pressure` diagnostic pass that walks every region
   and emits remarks with the (stub) results.
5. Verify on `bench/A/A001-single-scalar.mlir`: should emit zero-pressure
   remarks without crashing.

This proves the AnalysisManager integration and JSON config loading work
end-to-end before investing in the actual analysis logic.

---

## 15. Paper-ready framing

Title candidate: **"A Unified Analytical Cost Model for Cache and Register
Pressure in MLIR Loop Transformations."**

Story: MLIR's affine loop fusion explicitly documents its profitability
analysis as a "placeholder cost model" — a memory-footprint heuristic that
ignores both the cache hierarchy and the register file. We present a unified
analytical cost model that estimates per-decision cycle costs across the
memory hierarchy (L1/L2/L3/DRAM, with stride-aware access patterns) AND the
register file (per-class budget with target-specific spill cost). We replace
the placeholder in MLIR's affine loop fusion pass and evaluate on PolyBench.
On N kernels where the placeholder over-fuses (specifically when the fused
working set exceeds L1 or the combined live set exceeds the register
budget), our model correctly rejects the fusion, recovering up to 1.X×
performance. On M kernels where fusion is genuinely profitable, our model
matches the placeholder's decisions. The analytical framework is
target-pluggable via JSON configuration and supports x86-64 (AVX2,
AVX-512) and ARM Neon out of the box.

Contributions:
1. The unified cost model framework (architecture, dispatch, weight calibration)
2. The RegisterPressureAnalysis with three pluggable spill strategies and
   their calibration against LLVM regalloc on N programs
3. The replacement of MLIR's `--affine-loop-fusion` placeholder, with
   demonstrated PolyBench results
4. An evaluation methodology that distinguishes "avoiding regressions" from
   "creating speedups" — both matter, the literature conflates them

This is publishable in CC, CGO, or LCTES.

---

## 16. Phase 4.5: Affine loop tiling fork (weeks 6–7, inserted into plan)

**Locked-in addition** from middle-path decision. Tile-size selection is the
second pass we fork from upstream to drive with our cost model.

### 16.1 Setup

Create `lib/Transforms/AffineLoopTilingCostModel/`:
- Copy `mlir/lib/Dialect/Affine/Transforms/LoopTiling.cpp` (the current MLIR
  22 version)
- Add as `dr-affine-loop-tile` pass in `Passes.td`
- Track upstream SHA in `UPSTREAM.md` alongside the fusion fork

### 16.2 Replace `getTileSizes`

The placeholder at LoopTiling.cpp:99 picks tile sizes as `nth_root(footprint /
cacheSize)` — single cache level, uniform distribution across dimensions.
Replace with `UnifiedCostModel::decideTileSizes`.

### 16.3 `decideTileSizes` implementation

```cpp
TilingDecision UnifiedCostModel::decideTileSizes(ArrayRef<AffineForOp> band) {
  // Compute per-dimension reuse factor (how many times each loop's data
  // is accessed across iterations of outer loops). Heavily reused dims
  // get larger tiles.
  auto reuseFactors = computeReuseFactors(band);

  // For each candidate tile-size combination (limited to a search grid):
  //   T_i ∈ {1, 4, 8, 16, 32, 64, trip_count_i}
  TilingDecision best;
  best.totalCycles = UINT_MAX;

  for (auto tileSizes : generateCandidates(band, /*gridSize=*/6)) {
    // Working set check (must fit some cache level)
    int64_t workingSet = estimateTiledFootprint(band, tileSizes);
    unsigned cacheLevel = pickCacheLevel(workingSet, cache);

    // Register pressure check (tiled inner body's live set)
    auto innerRegion = synthesizeTiledInnerRegion(band, tileSizes);
    auto pressure = rpa.analyzeRegion(innerRegion);
    if (handler->checkConstraint(pressure, regs).has_value()) continue;

    // Cost
    unsigned memCycles = estimateTiledMemCycles(band, tileSizes, cacheLevel);
    unsigned regCycles = pressure.totalSpillCycles;
    unsigned aluCycles = baseAluCycles(band);  // unchanged by tiling
    unsigned total = combine(memCycles, regCycles, aluCycles);

    if (total < best.totalCycles) {
      best.tileSizes.assign(tileSizes.begin(), tileSizes.end());
      best.totalCycles = total;
      best.predictedSpills = pressure.totalSpillCycles / regs.spillReloadCycles;
      best.predictedFootprint = workingSet;
      best.rationale = ...;
    }
  }

  return best;
}
```

### 16.4 Evaluation extension

Add tiling configurations to the PolyBench script:
- **none:** no tiling
- **upstream-tile:** stock `--affine-loop-tile`
- **drcomp-tile:** `--dr-affine-loop-tile` with unified cost model
- **drcomp-tile+fuse:** both drcomp passes in sequence (the pipeline story)

Same hypothesis framework as Phase 5 (H1: avoids regressions, H2: preserves
wins, H3: predicted spills correlate, H4: weighted-sum beats hard-constraint).

### 16.5 Risk specific to tiling

Spike H showed that on Ryzen V-Cache (96MB L3) and Xeon Broadwell (45MB L3),
matmul kernels at PolyBench LARGE sizes (working sets ~25MB) fit in L3
comfortably — tiling is mostly noise. The contribution risk is that
`drcomp-tile` doesn't differ from `none` because no tile size matters.

**Mitigation:** Include EXTRALARGE size runs (working sets > L3) where
tiling MUST help (or hurt). Report results across sizes; if EXTRALARGE shows
signal that LARGE/STANDARD don't, frame the tiling contribution as
"effective on capacity-bound regimes."

### 16.6 Updated phase sequencing

| Phase | Topic | Weeks |
|-------|-------|-------|
| 1 | Architecture handler infrastructure | 1 |
| 2 | RegisterPressureAnalysis | 2–3 |
| 3 | DR integration | 4 |
| 4 | Affine fusion fork | 5 |
| 4.5 | Affine tiling fork (NEW) | 6 |
| 5 | PolyBench evaluation | 7–8 |
| **Total** | | **8 weeks** |

Tile-size selection adds 1 week net (overlap with fusion fork on shared
infrastructure: `UnifiedCostModel` API, synthetic region construction,
`analyzeHypothetical` machinery).
