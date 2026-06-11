# Cost-model architecture portability — audit (2026-06-08)

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

1. **`vl` is an element count, not ISA-derived** (`AffineRegisterBlock.cpp:850,
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
