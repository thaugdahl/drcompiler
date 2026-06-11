# Register Pressure Plan — Aggressive Path Extras

**Companion to `REGISTER_PRESSURE_PLAN.md`.**

The main plan executes the **middle path**: uniform-weight calibration,
3-class register model, FuncOp-scoped analysis, hypothetical-only cross-region
support. This document specifies the four upgrades that take you from middle
to **aggressive** if (a) the middle-path results are promising and you want
to strengthen the contribution, or (b) reviewers ask for more depth in any
of these specific areas.

**Treat each upgrade independently** — they have different cost/value
profiles and dependencies. Adopting all four roughly doubles the timeline
(8 weeks → 16 weeks).

---

## Upgrade A: Per-arch derived cost weights

### What it changes

**Middle path:** `(α_mem, β_reg, γ_alu) = (1.0, 1.0, 1.0)` as default; tune
via grid search on a held-out calibration set; report tuned values.

**Aggressive:** Per-arch weights derived from microarchitecture documentation
(Intel Optimization Reference Manual, Agner Fog tables, ARM TRM). The
derivation is part of the contribution — you publish a methodology for
extracting cycle weights from vendor docs and validating them empirically.

### Concrete deliverable

A new document `docs/CostWeightDerivation.md` per supported architecture:

```
=== x86-64-avx2 (Haswell baseline) ===
α_mem (memory cycle weight):  1.0
  Rationale: cache latency is the reference cycle.
β_reg (spill cycle weight):   1.3
  Derivation: spill reload from stack hits L1 (5 cycles) but also
  occupies an LSU port for 1 cycle; in a port-bound loop, the LSU
  occupancy carries an opportunity cost roughly equal to 30% of the
  reload cycle. Measured: 1.28 ± 0.04 on Skylake.
γ_alu (ALU cycle weight):     0.7
  Derivation: ALU ops on modern OOO cores execute on multiple ports
  in parallel. Average IPC for arith chains: 1.4. Effective cycle
  cost per op: 1/1.4 ≈ 0.7.

=== x86-64-avx512 ===
α_mem: 1.0
β_reg: 1.5  (mask spills go through kmov, expensive)
γ_alu: 0.6  (wider issue width)

=== arm-neon (Cortex-A78 baseline) ===
α_mem: 1.0
β_reg: 1.2
γ_alu: 0.8
```

### Implementation delta

- ~200 LOC across `lib/Analysis/ArchHandlers/*.cpp` to override `combineCosts`
  per arch with derived defaults
- ~400 LOC for a calibration validator (`scripts/validate_weights.py`)
  that runs known microbenchmarks and checks the derived weights match
  observed runtime within ±15%
- The new `docs/CostWeightDerivation.md` (~500 lines, including citations
  to vendor docs)

### Effort estimate

**~1 week.** Most of the work is the literature review and validation script,
not code.

### What it enables in the paper

- **Strongest claim:** "Our cost model uses cycle weights derived from
  vendor-published microarchitecture data, validated empirically." Reviewers
  cannot challenge "why these weights" — you cite Intel.
- **Per-arch ablation:** report results for each handler's derived weights AND
  uniform-weight baseline. Show that derived weights improve results by X%
  over uniform.

### Validation requirements

- Microbenchmark suite (memory-bound, register-bound, ALU-bound variants)
- Per-arch hardware access (you need Skylake or newer AVX-512 hardware to
  validate the AVX-512 weights — your existing Xeon Broadwell only has AVX2)
- Statistical methodology (confidence intervals on the derived weights)

### Risks

- **Vendor docs are wrong or incomplete.** Intel's published latencies
  don't always match observed behavior, especially for AVX-512.
- **Hardware coverage gap.** If you can't validate AVX-512 weights on
  AVX-512 hardware, you publish unvalidated numbers.
- **Reviewer attack:** "Your weights are calibrated to Haswell but evaluated
  on Skylake — show me the cross-microarchitecture sensitivity."

### When to revisit (upgrade trigger)

Take this upgrade if:
- Middle-path uniform-weight results are weak (drcomp doesn't clearly beat
  upstream) AND you suspect weight calibration is the issue
- A reviewer of the middle-path paper specifically asks for derived weights
- You get access to validation hardware for multiple architectures

Skip this upgrade if:
- Uniform weights + tuned calibration already produces a clean story
- You don't have multi-arch hardware (the upgrade's value is in
  per-arch differentiation)

---

## Upgrade B: Separate FP-scalar register class

### What it changes

**Middle path:** Three register classes (GP, Vec, Pred). Scalar `f32`/`f64`
classified as Vec on x86-64 (uses XMM).

**Aggressive:** Four register classes (GP, FP-scalar, Vec, Pred). Scalar
floats get their own budget and spill characteristics.

### Why it might matter

Some architectures genuinely have separate scalar FP register files:
- **MIPS, SPARC:** dedicated 32 FP regs separate from GP
- **RISC-V (F/D extensions):** 32 FP regs separate from GP
- **ARM AArch32 with VFP:** separate scalar FP register file

Even on x86-64 where scalar FP shares XMM with vector, there's a subtle
distinction: scalar FP ops use only one lane of an XMM, and the regalloc
treats them somewhat separately (different live range characteristics).

### Concrete differences in the model

```cpp
// Middle path (3 classes):
enum class RegClass { GP, Vec, Pred };

// On x86-64:
classifyType(f32) -> {Vec, 1}    // uses xmm
classifyType(f64) -> {Vec, 1}    // uses xmm
classifyType(vector<4xf32>) -> {Vec, 1}  // also xmm
// → all three compete for the same 16-reg budget

// Aggressive (4 classes):
enum class RegClass { GP, FpScalar, Vec, Pred };

// On x86-64:
classifyType(f32) -> {FpScalar, 1}
classifyType(f64) -> {FpScalar, 1}
classifyType(vector<4xf32>) -> {Vec, 1}
// → budget split: FpScalar gets ~8 regs, Vec gets ~8, both drawn from
//   the same physical xmm0-15 pool
```

The aggressive model captures that scalar FP and packed FP have different
lifetimes (scalar FP typically lives across function calls more often;
packed FP is loop-local).

### Implementation delta

- 1 new enum variant
- New per-arch `fpScalarBudget` parameter
- Updated `classifyType` in every arch handler (4 handlers)
- Updated `PressureResult` to track 4 classes
- Updated `SpillStrategy` implementations to aggregate over 4 classes
- ~200 LOC total

### Effort estimate

**~3 days.** Mostly mechanical.

### What it enables in the paper

- **Portability claim:** "Our model handles architectures with separate
  scalar FP register files (RISC-V, MIPS, ARM AArch32)." Concrete evidence
  of generality beyond x86-64.
- **More accurate predictions on x86-64:** in cases where the regalloc
  splits scalar vs packed differently (~5-10% of cases empirically).

### Validation requirements

- A few synthetic kernels with deliberate mixes of scalar FP and packed FP
- Verify the 4-class predictions match `llvm-mca` better than 3-class

### Risks

- **Marginal accuracy improvement on x86-64.** The 4-class model is more
  correct in theory but the regalloc heuristics may not align with the
  partition you predict. Worst case: 4-class is no more accurate than
  3-class and you've added complexity for nothing.
- **Calibration burden:** with 4 budgets per arch (instead of 3), you have
  more parameters to defend.

### When to revisit

Take this upgrade if:
- You add RISC-V or MIPS to the supported architectures (then it's
  required, not optional)
- Calibration with 3 classes shows systematic over/under-prediction in
  FP-heavy kernels (PolyBench `gemm`, `2mm`, etc.)
- A reviewer asks "what about architectures with separate FP register files?"

Skip this upgrade if:
- You're staying on x86-64 + ARM Neon (Neon also unifies scalar+vec)
- Middle-path calibration is already accurate enough

---

## Upgrade C: Cross-region pressure modeling

### What it changes

**Middle path:** Pressure analysis stays within a single Region (loop body,
function body). The exception is `analyzeHypothetical` for DR's
interprocedural remat — which builds a synthetic merged region on demand.

**Aggressive:** General interprocedural pressure analysis. Pressure at a
call site accounts for callee-saved register usage, caller-saved spill
costs at the call point, and chains of calls.

### Why it might matter

Function calls are clobber points: caller-saved registers (typically half
the budget on x86-64) must be spilled by the caller before the call. The
ABI defines which registers survive.

Without cross-region modeling:
- Pressure analysis of a caller treats a call as "magic" — doesn't know
  the cost of saving live caller-saved regs across it
- A call point in the middle of a hot loop is invisible pressure-wise
  even though it may force spills

With cross-region modeling:
- Each call is annotated with "live values that must be saved" = `live ∩ caller_saved`
- Saved register cost = `count × (spillStore + spillReload)` cycles per call
- Long-chained calls amplify the effect

### Concrete example

```cpp
// Hot loop with a function call in the middle
affine.for %i = 0 to N {
  %a = arith.addf %v1, %v2 : f64
  %b = arith.mulf %v3, %v4 : f64       // 6 live values: v1-v4, a, b
  func.call @sigmoid(%a) : (f64) -> f64  // CALL POINT
  %c = arith.addf %b, %d : f64
  ...
}
```

Middle path: pressure analysis sees 6 live values, all f64 → Vec class,
budget = 16 → no pressure → 0 spills predicted.

Aggressive: at the call point, the live set `{v1, v2, v3, v4, b, d}` must
be saved across the call (assuming they're in caller-saved regs). With
6 live × 5cy spill reload × N trip count, expect substantial added cost.

### Implementation delta

- New `CallClobberAnalysis` — given an ABI, identify caller-saved vs
  callee-saved registers
- ABI table per supported triple
- Modified `RegisterPressureAnalysis` to walk through call points and
  add save/restore cost
- `analyzeRegion` now traverses across function boundaries via the call
  graph (bounded depth to avoid blowup)
- ~600 LOC

### Coupled with Upgrade D

This upgrade **requires** `ModuleOp`-scoped analysis (Upgrade D below) to
work cleanly. AnalysisManager at FuncOp scope can't safely query across
functions. So adopting Upgrade C forces you to also adopt Upgrade D.

### Effort estimate

**~2 weeks** (plus Upgrade D's overhead — see below).

### What it enables in the paper

- **Broader applicability:** interprocedural pressure is critical for
  programs with many small functions (template-heavy C++, language
  runtimes, ONNX models lowered with per-op function dispatch)
- **DR-specific synergy:** strengthens the interprocedural remat strategy
  (Strategy 4) by making the cost of cross-function value materialization
  visible

### Validation requirements

- Test programs with calls in hot loops
- Compare predicted save/restore costs to LLVM-emitted save/restore
  instruction counts (the calling convention enforces this — should match
  exactly, not approximately)
- ABI correctness for each supported target

### Risks

- **Call graph cycles:** recursive functions need cycle-breaking in the
  analysis traversal
- **Indirect calls:** function pointers / virtual calls can't be resolved
  statically. Fall back to conservative assumption (assume worst-case
  ABI clobber).
- **External function calls:** for libcalls (printf, malloc), assume
  full clobber. Documented assumption.

### When to revisit

Take this upgrade if:
- Your evaluation expands to ONNX models or C++ template-heavy workloads
  where call density in hot loops is high
- DR's interprocedural strategy results suggest pressure underestimation
  (e.g., predicted "no spills" but measured spills increase after
  interprocedural remat)
- A reviewer specifically asks "what about function calls?"

Skip this upgrade if:
- Your evaluation is PolyBench (mostly single-function kernels) and
  drcompiler's interprocedural remat isn't being aggressively evaluated

---

## Upgrade D: ModuleOp analysis scope

### What it changes

**Middle path:** `RegisterPressureAnalysis` is anchored on `func::FuncOp`.
Each function's analysis is cached and invalidated independently.

**Aggressive:** Analysis anchored on `ModuleOp`. One module-wide analysis
that knows about every function, supports cross-function queries.

### Why it might matter

Couples with Upgrade C. Cross-region pressure needs to inspect callees, and
AnalysisManager doesn't support cleanly querying "analysis for function X"
from a pass running on function Y if both are FuncOp-scoped. ModuleOp scope
makes this trivial.

### Concrete implementation differences

```cpp
// FuncOp scope (middle):
class RegisterPressureAnalysis {
  RegisterPressureAnalysis(FuncOp func, ...);
  PressureResult analyzeRegion(Region &);  // Only regions within `func`
};
// In a pass:
auto &rpa = getAnalysis<RegisterPressureAnalysis>();  // for current func

// ModuleOp scope (aggressive):
class RegisterPressureAnalysis {
  RegisterPressureAnalysis(ModuleOp mod, ...);
  PressureResult analyzeRegion(Region &);   // Any region in any function
  PressureResult analyzeAtCallSite(CallOp); // New API
};
// In a pass:
auto &rpa = getAnalysis<RegisterPressureAnalysis>();  // for whole module
// Can query any function's pressure from anywhere
```

### Implementation delta

- Change `getAnalysis<>()` registration scope
- Update all `PassRegistration` for passes that consume the analysis
- The analysis itself: lazy per-function computation with module-wide cache
- Invalidation: when any function changes, that function's per-function
  cache invalidates, but the module-wide structure (call graph, ABI
  context) stays valid
- ~200 LOC for the wrapper, plus invalidation handling

### Performance implication

ModuleOp-scoped analysis can be more expensive per query if it eagerly
computes for all functions. Mitigate with lazy per-function computation
(compute only when first queried).

### Effort estimate

**~3 days** standalone, but only useful in conjunction with Upgrade C.

### What it enables in the paper

(Inherits from Upgrade C — see above.) Additionally:
- **Cleaner architecture:** the unified cost model can answer queries
  about any function from anywhere, matching the "unified framework"
  framing.

### Validation requirements

- Regression tests: existing FuncOp-scope tests still pass after upgrading
- Invalidation correctness: modifying function A shouldn't break cached
  results for function B

### Risks

- **AnalysisManager overhead:** if many passes don't need cross-function
  queries, paying ModuleOp-scope overhead is wasted
- **Subtle bugs in invalidation:** module-scoped invalidation is harder to
  reason about

### When to revisit

Take this upgrade if and only if you take Upgrade C. Otherwise it's
strictly worse than FuncOp scope.

---

## Combined adoption: cost and value

If you adopt all four aggressive upgrades:

| Upgrade | Time | Independent value | Combined value |
|---------|------|-------------------|----------------|
| A: Per-arch weights | 1 week | High (stronger calibration claim) | Synergy with C: per-arch weights become more important when you're crossing call boundaries |
| B: FP-scalar class | 3 days | Low on x86-64; required for RISC-V | Independent |
| C: Cross-region | 2 weeks | Medium (only if call-heavy workloads) | Requires D |
| D: ModuleOp scope | 3 days | None standalone | Required by C |
| **Total** | **~4 weeks** | | |

**Combined timeline:** Middle path = 8 weeks. Middle + all aggressive
upgrades = ~12 weeks (some overlap; not strictly additive).

---

## Decision framework: when to upgrade mid-project

Don't commit upfront. Run the middle path first. Make upgrade decisions at
defined checkpoints based on observed results:

### Checkpoint 1: After Phase 2 (RegisterPressureAnalysis validated)

**Question:** Does Pearson r between predicted spills and `llvm-mca` measured
spills hit > 0.85 on the 20-program calibration set?

- **Yes:** middle path is on track. Continue.
- **No, r ∈ [0.6, 0.85]:** consider Upgrade B (FP-scalar separation) to
  improve accuracy
- **No, r < 0.6:** the 3-class model may be too coarse; consider Upgrade B
  AND look hard at the Spill strategy choice

### Checkpoint 2: After Phase 3 (DR integration tested)

**Question:** Does the augmented cost model change DR decisions on the
existing 202-test bench suite in expected ways?

- **All decisions match middle-path predictions:** continue
- **Some decisions diverge unexpectedly, especially for interprocedural
  cases:** consider Upgrade C (cross-region) to capture missed pressure

### Checkpoint 3: After Phase 5 (PolyBench evaluation)

**Question:** Does drcomp clearly beat upstream on the PolyBench evaluation?

- **Clear win across ≥3 kernels:** middle path delivered. No upgrades
  needed unless reviewers demand them.
- **Mixed results, sensitivity to weights is high:** consider Upgrade A
  (derived weights) for the next iteration
- **No clear win, results are noise:** the fundamental approach may not
  bite on PolyBench. Pivot evaluation target (try ONNX models, or
  C++/template-heavy benchmarks) before adding aggressive upgrades

### Reviewer-response upgrades

Some upgrades are best done in response to specific reviewer asks:

- "Why uniform weights?" → Upgrade A
- "What about RISC-V?" → Upgrade B
- "What about programs with many function calls?" → Upgrades C + D

Don't preemptively add complexity for hypothetical reviewers; let the
actual feedback drive the upgrade decision.

---

## Anti-recommendations

Things the aggressive path does **not** propose, even though they might
seem obvious:

- **Branch prediction modeling.** Out of scope. Use llvm-mca for this if
  needed; don't bake into the cost model.
- **SMT/Z3-based optimal regalloc prediction.** Too slow for compile-time
  use. Stick to graph coloring approximation.
- **Learned cost models (MLP/transformer).** Defeats the analytical
  framing. Save for a separate paper.
- **Whole-program partition-aware analysis.** Way out of scope; would
  require dataflow-style analysis across the entire module. Use case
  too narrow to justify.

---

## Summary

The middle path delivers a complete, defensible contribution. The aggressive
upgrades each address a specific weakness or reviewer concern. Adopt them
**reactively, not preemptively** — based on what the middle-path checkpoints
reveal.

If forced to recommend one upgrade to adopt eagerly: **Upgrade A (per-arch
weights)**, because the methodology contribution (deriving weights from
vendor docs) is publishable independently and strengthens the calibration
defense regardless of the specific evaluation outcome.
