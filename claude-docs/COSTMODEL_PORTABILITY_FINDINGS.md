# Cost-model architecture portability — audit (2026-06-08)

---

## Portability cleanup (2026-07-27): gaps #2, #4, #5, #7 closed

Everything the audit below flagged as a hardcoded literal or a missing handler
is now resolved.  What remains is *validation*, not plumbing.

**Cache geometry (#2, #5)** — landed earlier as `costmodel_p0d/e/f`
(CROSSCUTTING.md III.1): fission, the tiler's reuse analysis, and the LoopFusion
fork all read `cacheLine` / `memLatency` / `llcSharers` from `MachineModel`, and
the fusion fork's stale `{.,262144,0,.}` literal is now the unified Zen4 geometry
`{32768, 1048576, 33554432, 4, 12, 40, 200, 64}`.  NEW here: fission also gained
the two CLI overrides it lacked (`mem-latency`, `cache-line-size`), so every
geometry field it consumes follows the full CLI > JSON > built-in contract
instead of JSON > default.

**Issue width (#4)** — `kIssueWidth = 4` in `CacheCostModel.cpp` is gone.  The
throughput divisor is `ArchParams::issueWidth`: a per-handler default,
overridable from the JSON `arch.issue_width`, threaded to all three
`estimateComputeCost` call sites (`BufferElim.cpp`, `Strategies/PartialRemat.cpp`,
`DataRecomputation.cpp`) from the ArchParams each already had in scope.  Clamped
to >= 1 so a malformed `issue_width: 0` cannot divide by zero.

Byte-identical on every target the paper reports: `generic`, `x86-64-avx2`,
`x86-64-avx512`, and `arm-neon` all default to 4 — the literal's value.  Zen4
dispatches 6 macro-ops and raising the x86 default is defensible, but it
re-prices every keep-vs-recompute decision, so it needs its own validation run
rather than a silent default bump.  `apple-m-series` (8) is the first handler
that differs.

**Handlers (#7)** — two new handlers behind the existing interface, no new pass:

| handler | vec | issue | gp/fp/vec/pred | selected by |
|---|---|---|---|---|
| `apple-m-series` | 128 (NEON) | 8 | 31/32/32/0 | `aarch64-*-darwin` triple, or by name |
| `arm-sve` | 128 (VL floor) | 4 | 31/32/32/16 | **name only** |

`pickHandlerForTriple` routes aarch64-darwin to `apple-m-series`; SVE is a
feature, not an architecture, so no triple implies it.  The Apple 128 B cache
line is deliberately NOT in the handler — it is memory geometry and belongs in a
probed `cache.cache_line: 128` JSON (one machine, one description).  AMX is not
modeled: the code generator never targets it.

SVE models the 128-bit architectural VL *floor*.  That direction is the safe
one: register pressure computed at VL=128 is never under-estimated on a wider
implementation, so an accepted candidate stays register-legal.  Vector
throughput, on the other hand, is under-estimated on a 256/512-bit part — model
that by pinning `arch.vector_width_bits` (and `vector_bits_native`/`_arch` for
register blocking) to the deployment target's VL.

**Tests** — `test/Analysis/PrintArchHandler/{apple-m,triple-darwin,sve,issue-width-json}.mlir`
(the last one pins JSON-over-handler-default precedence: apple-m-series' 8
overridden to 6).  The eight pre-existing expectations gained the `issue=` field.

**Handler vs MachineModel split (know this before running on ARM):** a handler
supplies register budgets, classification, combiner weights, and issue width.  It
does NOT feed `MachineModel`'s vector-execution fields — `vectorBitsNative` /
`vectorBitsArch` / `vecRegBudget` still come from the JSON `arch` block, and
without them `hasExplicitVectorModel` stays false, so register blocking keeps the
static `vl` default (8, the Zen4 value) even under `apple-m-series`.  A real
Apple-M or SVE run therefore needs a JSON that sets the vector model (128/128 for
NEON-class parts) and the cache block (128 B line on Apple), not just
`arch.handler`.  Unifying those two sources is the deferred register-budget merge
in CROSSCUTTING.md III.1 — it changes the VL decision, so it is not a refactor.

**Still open (hardware, not plumbing):**
- No handler here has been run on the hardware it describes.  They are
  structurally complete and unvalidated — same status `arm-neon` always had.
- `mr`/`nr` are still not arch-derived, so the NEON accumulator-oversizing note
  in gap #1 below stands for `apple-m-series` and `arm-sve` too.
- `kIssueWidth`'s successor is a single scalar; a real front end has per-port
  limits (FP vs load/store vs branch).  Only wide independent cones are
  sensitive, so this is a modeling choice, not a gap.

---

## WP-G1 update (2026-06-12): `vl` IS now ISA-derived — gap #1 resolved

`ONNX_O3_GAP_STEPS.md` WP-G1.  The register-block vector width is no longer a
hardcoded element count: `MachineModel` carries a **vector-execution model**
(`vectorBitsNative`, `vectorBitsArch`, `vecRegBudget`, `avx512FreqThrottle`) and
`preferredVectorElems(elemBytes, mr, nr)` derives `vl`.  The pass derives it
when a cost-model JSON describes the machine (`arch.vector_bits_native` etc.)
and the CLI did not pin `--vl`; the default (no JSON) keeps the static option
default 8 — which is exactly what the model derives for this Zen4 host — so
every pre-WP-G1 lit test is byte-identical (verified: AffineRegisterBlock 22/22,
full suite 215/0).  New guard: `test/AffineRegisterBlock/vl-from-machine-model.mlir`
(DEFAULT→v8, XEON JSON→v16, `--vl 8` override→v8).

### The model (reproduces every measured point)

`vl = nativeElems`, raised (halving the accumulator count) until the
`mr·⌈nr/vl⌉` accumulator tile fits `vecRegBudget`, capped at the encodable
width.  `nativeElems = vectorBitsNative / elemBits`.

| machine | native | f32 vl | f64 vl | basis |
|---|---|---|---|---|
| Zen4 (this host) | 256 | **8** | **8** (4→reg-floor 8) | == old flat default |
| Xeon Gold (Idun, planned) | 512 | **16** | 8 | native 512 datapath |
| NEON | 128 | 4 | 2 | (needs mr/nr arch-derivation too) |

### Reconciliation with spike #4 (below) — NOT a contradiction

Spike #4 measured f32 GEMM N=1024 on **this Zen4 host**: vl=8 (88 GFLOPs) beat
vl=16 (77, −13%) — "filling the register is wrong."  That is a **native<arch**
phenomenon: on Zen4 a 512-bit zmm-FMA is double-pumped (2 µops over the 2×256
FP pipes) so ymm(8) and zmm(16) have **equal peak FLOPs**, and the extra ILP of
16 accumulators (vl=8) vs 8 (vl=16) wins.  The model encodes exactly this via
`vectorBitsNative=256`.  On a **native-512** Xeon, zmm has **2× the peak** of
ymm (not double-pumped), so vl=8 would halve throughput — there vl=16 wins, and
the model gives 16.  Spike #4's rule is the Zen4 special case of the general
native-width rule, not a universal "8 is best."

### resnet50 end-to-end attribution (2026-06-12, Zen4, median-of-9, back-to-back)

The `--vl 8` end-to-end win over `--vl 16` (1.226 vs 1.526 s) is **NOT** the
GEMMs.  Cross-build (conv-vl decoupled from GEMM vl):

| config | median (s) | speedup vs vl16 (s, + = faster) |
|---|---|---|
| vl16 (conv@16, GEMM@16) | 1.526 | — |
| GEMM→8, conv@16 | 1.534 | **−0.008 (noise — GEMMs indifferent)** |
| conv→8, GEMM@16 | 1.213 | **+0.313 (the whole win — conv coverage)** |
| vl8 (both) | 1.226 | +0.300 |

The entire win is the 5× 14×14 conv interiors (spatial extent 8) clearing the
`interiorWidth ≥ VL` gate at VL=8 — a **coverage** effect handled per-band by
WP-G3's `pickVL(extent)`, orthogonal to the machine VL.  GEMMs are
VL-indifferent on Zen4 (peak-FLOP-bound, both widths fit registers), as the
model predicts.

### avx512FreqThrottle + the Idun spike (planned)

`avx512FreqThrottle` (Intel AVX-512 license downclock, ~0.85–0.90; **value =1.0
inert on Zen4 — no license**) is a documented portability field.  Its VALUE does
NOT flip the VL decision on either Zen4 (native<arch) or a native-512 Intel part
(real throttle ≥0.85 ≫ 0.5, so wide never loses to half-width) — it feeds
cross-machine throughput prediction.  Note: its mere *presence* in a JSON does
opt that JSON into model-derived VL (it sets `hasExplicitVectorModel`), so a
machine file that wants the default vl but records the throttle for prediction
should also set `vector_bits_native`/`_arch` explicitly to the intended values.

**Idun spike (56-core Intel Xeon Gold):** measure (1) the actual sustained
512-bit FMA throttle (`perf` core-freq under a vl=16 GEMM), and (2) whether the
*measured* f32 GEMM optimum is vl=16 (model prediction) or whether a severe
throttle pulls it to vl=8.  Feed the result as `arch.vector_bits_native` /
`avx512_freq_throttle` in an `idun-xeon.json` and rerun the resnet50 bench under
`--cpu-cost-model-file`.  Expected: vl=16 wins on Idun (native 512); if not, set
`vectorBitsNative=256` for that part and document the throttle threshold.  This
is the single open validation of the portable model on a non-Zen4 part.

---


Scope: how portable is the drcompiler cost model off x86/Zen4 (AArch64 Neoverse,
Apple M-series, ARM SVE)? Verified against the source (file:line cited); two
claims from the first-pass audit were overstated and are corrected here.

## Verdict

**The decision machinery is arch-parametric where it matters most; the gaps are
in (a) a few hardcoded `cacheLineSize`/`memLatency` literals outside the
DataRecomputation pass, (b) `vl` not being derived from the target ISA, and (c)
no Apple-M / SVE handlers.** The default is arch-neutral (`generic`), not an x86
profile, so nothing is silently mis-tuned for x86 — it is simply untuned until a
handler/JSON is supplied.

## What IS parameterized (portable)

- **Arch handler selection** (`ArchHandlerRegistry.cpp:46`): `dr-arch-handler`
  CLI name > JSON `handler` > JSON `triplet`→`pickHandlerForTriple` > **default
  `"generic"`**. `pickHandlerForTriple`: x86_64→`x86-64-avx2`,
  aarch64→`arm-neon`, else `generic`. Handlers available: `generic` (128b,
  gp/fp/vec=16), `x86-64-avx2` (256b), `x86-64-avx512` (512b, fp/vec=32,
  pred=8), `arm-neon` (128b, gp=31, fp/vec=32). Register budgets, vector width,
  and combine weights (α/β/γ) are all per-handler + JSON-overridable.
  - CORRECTION to first-pass audit: the *default* is `generic`, NOT
    `x86-64-avx2`. Triple-based selection only fires when a triple is supplied.
    There is no host auto-detect.
- **DataRecomputation cache geometry**: fully option-wired —
  `dr-l1/l2/l3-size`, `dr-l1/l2/l3/mem-latency`, **`dr-cache-line-size`**,
  `dr-llc-sharers`. So the spike-3 stride-aware footprint reads
  `cache.cacheLineSize` from an option and re-prices per arch. **Verified**:
  same gather kernel, `dr-cache-line-size=64→128` moves the footprint
  262148→393220 (the 96 B stride is line-capped at 64, then admitted at 128) —
  set 128 for Apple-M and the model is correct.
- **MemoryFission** geometry: l1/l2/l3 size + latencies + `llc-sharers` +
  `l2-occupancy-pct` are options (post spike-2 unification).
- **AffineLoopTiling**: latencies + sizes read from JSON (`orElse(...)`).

## What is HARDCODED x86/Zen4 (gaps, by impact)

1. **`vl` is an element count, not ISA-derived** — **RESOLVED 2026-06-12, see
   the WP-G1 update at the top of this file.**  `vl` now derives from
   `MachineModel::preferredVectorElems` (native datapath width + register-
   pressure floor), JSON-gated so the Zen4 default is unchanged.  The NEON
   mr/nr-oversizing note below still stands (needs ARM hardware).  Original
   finding kept for the record: (`AffineRegisterBlock.cpp:850,
   978`: `VectorType::get({(int64_t)VL}, elemTy)`; option default 8). On NEON
   (128 b) `vl=8` fp64 = 512 b = 4 q-regs/vector → the mr×nr tile over-subscribes
   the 32×128b file; SVE scalable width is inexpressible at compile time.
   **SPIKE #4 RESULT (measured, overturns the "fp32 half-zmm bug"):** the
   supposed bug — fp32 `vl=8` = 256 b = "half a zmm" — is NOT a perf bug. f32
   GEMM N=1024 on AVX-512 (core 8, 3×median-of-9, checksum MATCH): **vl=8 ≈ 88
   GFLOPs, vl=16 (full zmm) ≈ 77 (−13%), vl=4 ≈ 55.** The broadcast µkernel is
   ILP-bound, not register-width-bound: vl=8 keeps mr×⌈nr/vl⌉ = 16 independent
   accumulators (backend uses ymm); vl=16 halves them to 8 → ILP-starved. So
   filling the register is the WRONG default. Added a `vector-width-bits` option
   + `vl=0` auto (vl = width/elemBits) as an OPT-IN scaffold for
   register-constrained targets (IR-verified: width=128 → f32 v4 / f64 v2), but
   it is measured-slower on AVX-512 and stays opt-in; default vl=8 unchanged.
   True NEON/SVE portability also needs mr/nr arch-derivation (the mr·nr=128
   accumulator values exceed NEON's 512 B register file at any vl) — needs ARM
   hardware to tune, not spikeable on x86.
2. **`cacheLineSize = 64` hardcoded** in `MemoryFission.cpp:375`,
   `AffineLoopTilingCostModel/LoopTiling.cpp:264`,
   `AffineLoopFusionCostModel/LoopFusion.cpp:77` (DR is fine — it uses the
   option). Wrong on Apple M-series (128 B). Low functional impact today:
   fission's decision uses buffer-size `bufLat`, not the stride path, so the
   line size barely moves it — but it WILL matter if the shared stride-aware
   footprint is ever consulted there.
3. **`AffineLoopFusion` fork uses STALE geometry** (`LoopFusion.cpp:77`):
   `{32768, 262144, 0, 4, 12, 40, 200, 64}` — l2 = 256 KB (4× too small for
   Zen4; spike-2 set DR/fission to 1 MB) and l3 = 0 (contention machinery dead
   here). This is a staleness bug, not just non-portability: it mis-decides on
   *every* arch including x86. Highest-value fix.
4. **`kIssueWidth = 4` hardcoded** (`CacheCostModel.cpp:32`). Generic-superscalar
   floor; only bites wide independent expressions (critical path usually
   dominates). Apple M (~6-8 wide) and Neoverse N1 (~3) differ but the error is
   bounded.
5. **`memLatency = 200` hardcoded** in the fission/tiling `CacheParams` literals
   (DR has the option). 
6. **`elemBits = 64` (f64) working assumption** in tiling
   (`LoopTiling.cpp:287`).
7. **No Apple-M handler** (aarch64-darwin → `arm-neon`, loses the 128 B line)
   and **no SVE handler** (scalable vectors can't be modeled; falls to generic).

## Minimal portability fix list (prioritized, not yet done)

1. Fix the stale `LoopFusion` geometry literal → match the unified Zen4 geometry
   / thread the same options (correctness on all archs).
2. Thread `cacheLineSize` + `memLatency` as options into MemoryFission &
   tiling (or source from the handler) — removes the 3 hardcoded 64 s.
3. Derive `vl` from `ArchParams::vectorWidthBits / elemBits` (spike #4); add an
   `effectiveVectorLaneCount(elemTy)` to the handler. Fixes the fp32 half-zmm
   and NEON oversizing.
4. Move `kIssueWidth` into `ArchParams.issueWidth` (thread through
   `estimateComputeCost`).
5. Add `apple-m-series` (128 B line, NEON, ~6-wide) and a conservative
   `arm-sve` (pred budget 16, vl floor 128 b) handler; route
   aarch64-darwin → apple-m-series.

## Bottom line

The path that the spike-3 work touches (DataRecomputation footprint) is fully
arch-overridable and was verified to honor a non-default cache line. The
portability debt is concentrated in the codegen `vl` (spike #4) and in a handful
of hardcoded literals in the fission/tiling/fusion forks — none of which the DR
decision path depends on. The LoopFusion stale-geometry literal is the one item
that is wrong *today on x86 too* and should be fixed regardless of portability.
