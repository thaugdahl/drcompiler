# DR Runtime Benchmark Extensions

Status quo: 14 runtime benchmarks producing real speedups.
- **MemoryFission**: 2.2–3.0x (6 benchmarks, proven)
- **Partial remat**: 1.4–2.2x (8 benchmarks, proven)

This document specifies additional benchmarks to broaden DR coverage.
Each entry states the pattern, which strategy fires, why LLVM cannot
replicate it, and the expected speedup range.

---

## What LLVM -O2 already handles (don't bother)

| Pattern | LLVM pass | Notes |
|---------|-----------|-------|
| Same-block store→load | mem2reg / SROA | Even through alloca |
| Scalar global store→load in same function | GVN after inlining | Private functions get inlined |
| Loop-invariant load hoisting | LICM | |
| Redundant load elimination | GVN / EarlyCSE | |
| Dead store elimination | DSE | |
| Small-function inlining | Inliner | Private + small = always inlined |
| alloca buffer elimination | SROA | Splits alloca into SSA values |

## What LLVM cannot do (target these)

1. **Cross-loop recomputation with IV substitution** — producer loop
   stores `f(i)` to buffer, consumer reads `buf[j]`. DR partial-remat
   maps producer IV to consumer IV and clones `f(j)` inline.
   LLVM has no cross-loop value forwarding.

2. **Loop fission for redundant computation** — N sibling loops with
   identical expensive subexpression. LLVM has no loop fission pass.

3. **Interprocedural store→load forwarding through globals** when functions
   are non-inlinable (weak linkage, large bodies, or many call sites).

4. **Single-iteration extraction from writer loop** (Strategy F.1) —
   writer fills array in a loop, reader loads one element. DR extracts
   one iteration's body with IV replaced by the reader's index.

5. **Chained interprocedural materialization** (Strategy D) — value flows
   through 2+ globals via 3+ functions. DR builds a recursive sub-plan.

6. **Whole-buffer elimination** after all loads are replaced — removes
   malloc, all stores, and free. LLVM cannot eliminate malloc-backed
   buffers with cross-loop store/load patterns.

---

## Extension E1: Partial Remat — Strided Consumer Access

**Pattern.** Producer: `buf[i] = lut[i % K] * scale`. Consumer reads
`buf[i * S]` with stride S > 1. Each consumer access touches a fresh
cache line.

**Why it matters.** Strided access amplifies the effective memory cost.
A stride-16 f64 access loads 128 bytes per element (one full cache line)
but uses only 8. The cost model prices this via
`effBytes = min(cacheLineSize, stride * elemSize)`.

**Strategy.** Partial remat (S2b). The leaf `lut[...]` is L1-resident.

**Why LLVM can't.** Cross-loop IV substitution with stride transformation.

**Expected speedup.** 1.5–3x depending on stride and buffer size.

```
Variants:
  E1a: stride 4,  buf 16MB, lut 16   → ~1.5x
  E1b: stride 16, buf 16MB, lut 16   → ~2.0x
  E1c: stride 64, buf 64MB, lut 16   → ~2.5x
```

---

## Extension E2: Partial Remat — Multi-Leaf with Shared Subexpression

**Pattern.** Producer: `buf[i] = lut_a[i%8] * lut_b[i%8] + lut_c[i%4]`.
Three leaf loads from three small buffers. Consumer reads `buf[i]`.

**Why it matters.** Tests `dr-partial-max-leaves` budget (default 4).
Multiple leaf loads increase the recomputation ALU cost but each leaf
is L1-resident.

**Strategy.** Partial remat (S2b) with 3 leaves.

**Prerequisite.** The `pr_3leaves` pattern currently causes a dominance
violation in DR. Fix needed in `DataRecomputation.cpp` remat kernel —
the cloned leaf loads must be placed correctly when the producer's
`affine_map` mod expression is rematerialized at the consumer site.

**Expected speedup.** 1.3–1.8x (more ALU work per element dilutes the
memory bandwidth win).

```
Variants:
  E2a: 3 leaves, 16MB buf, lut sizes 8/8/4    → ~1.5x
  E2b: 2 leaves, 64MB buf, lut sizes 16/16    → ~1.8x (if dominance fix lands)
```

---

## Extension E3: Partial Remat — 2D Buffer with Tiled Leaf

**Pattern.** Producer fills a 2D buffer:
`buf[i][j] = weights[i % R][j % C] * scale + bias`
where `weights` is a small R×C tile (e.g. 8×8 = 512 bytes).
Consumer iterates over the full 2D buffer.

**Why it matters.** 2D access patterns are common in convolution and
stencil kernels. Tests whether partial remat handles multi-dimensional
affine maps and mod expressions correctly.

**Strategy.** Partial remat with 2D affine maps.

**Expected speedup.** 1.4–2.0x (same mechanism as 1D but with 2D
iteration).

```
Variants:
  E3a: buf 1024×1024 (8MB), weights 8×8   → ~1.6x
  E3b: buf 2048×2048 (32MB), weights 16×16 → ~2.0x
```

---

## Extension E4: Cross-Function F.1 Extraction at Scale

**Pattern.** Writer fills a large global array via loop:
`g[i] = base * i + offset`. Reader loads `g[idx]` where `idx` varies
per call. Caller calls writer once, then reader in a tight loop with
different indices.

**Why it matters.** Strategy F.1 extracts one iteration of the writer
loop and substitutes the reader's index for the loop IV. This
eliminates the global array load entirely — the reader gets the
inlined expression `base * idx + offset`.

**Strategy.** CrossFnOrdered, sub-strategy F.1.

**Why LLVM can't.** The writer fills an entire array; the reader loads
a single element. LLVM cannot extract a single loop iteration and
substitute the index argument. Even after inlining, GVN sees a store
loop and a single load — it cannot connect them without provenance
analysis.

**Defeating LLVM inlining.** Use `llvm.linkage = weak` on writer
(it contains a loop — cost model disfavors inlining loops). Or make
the writer body large enough (>100 instructions).

**Expected speedup.** 10–50% (eliminates one global load per reader
call; the load is L1-hot if repeated, but with large arrays it may
be L2/DRAM-hot depending on access pattern).

```
Variants:
  E4a: array 64K, reader accesses sequentially    → ~1.05x (L1-hot)
  E4b: array 64K, reader accesses randomly         → ~1.2x (cache-miss-prone)
  E4c: array 1M, reader accesses sequentially      → ~1.1x (L2/L3)
  E4d: array 1M, reader accesses with stride 128   → ~1.3x (cache-line waste)
```

---

## Extension E5: Cross-Function Chained Materialization (Strategy D)

**Pattern.** Three-function chain through two globals:
```
writer_a(x)  → stores x*x+1 to @ga
writer_b()   → loads @ga, stores ga_val*2+3 to @gb
reader()     → loads @gb
caller: writer_a(x); writer_b(); reader()
```
DR builds a depth-2 sub-plan: materializes `(x*x+1)*2+3` at the caller.

**Why it matters.** Tests the recursive plan-builder in Strategy D.
Multi-hop interprocedural chains appear in real code when helper
functions compose transformations through shared state.

**Strategy.** CrossFnOrdered, sub-strategy D.

**Why LLVM can't.** Three function calls through two globals.
Even with inlining, GVN must forward store→load through two separate
globals with intervening calls. LLVM's MemorySSA does not track
cross-global forwarding chains.

**Defeating LLVM.** Writer functions have `weak` linkage. Or make
bodies contain inner loops so inlining cost exceeds threshold.

**Expected speedup.** 5–15% on scalar chains (globals are L1-hot).
Larger with array globals where loads are more expensive.

```
Variants:
  E5a: scalar globals, 50M iterations, cheap ALU    → ~1.05x
  E5b: scalar globals, weak linkage, 50M iterations → ~1.10x (if not inlined)
  E5c: array globals (1K elements), random reader    → ~1.15x
```

---

## Extension E6: Buffer Elimination — Heap Buffer with Cross-Loop Pattern

**Pattern.** Allocate a heap buffer, fill it in loop A, consume in
loop B, deallocate. DR replaces all consumer loads (full-remat or
partial-remat), then buffer-elim erases the entire alloc+fill+dealloc
chain.

**Why it matters.** LLVM's SROA only handles alloca, not malloc.
After lowering, `memref.alloc` becomes `malloc`. If DR can replace
all loads and erase the buffer, it eliminates the malloc/free overhead
AND the producer loop (which becomes dead stores).

**Strategy.** DirectForward or PartialRemat → BufferElim.

**Why LLVM can't.** malloc-backed buffers with stores in one loop
and loads in another are opaque to SROA. DSE won't remove stores
because the loads consume them (from LLVM's perspective, since it
can't prove they're replaced by DR's recomputation).

**Prerequisite.** `dr-erase-eliminated-buffers=true` must be set, and
all loads from the buffer must be successfully replaced before
buffer-elim fires.

**Expected speedup.** 10–30% from eliminating malloc/free + dead
producer loop. Higher for small buffers where allocation overhead
is a significant fraction of runtime.

```
Variants:
  E6a: scalar buffer, 100M iterations, cheap ALU → ~1.10x
  E6b: 1K buffer, 100K iterations, partial-remat leaves → ~1.20x
  E6c: 64K buffer, 1K iterations, full alloc lifecycle per iter → ~1.15x
```

---

## Extension E7: Fission + DR Combined Pipeline

**Pattern.** Two sibling loops with identical `sqrt(x^2+1)/(x+eps)`
chain from a source buffer that was itself produced by a cross-function
writer via a global. DR eliminates the global store→load; fission
then operates on the remaining duplicate chains.

**Why it matters.** Tests pass composability. In the full compilation
pipeline (`data-recomputation` then `memory-fission`), DR simplifies
the IR before fission analyzes it. The combined effect may exceed
either pass alone.

**Strategy.** CrossFnOrdered (S4) + MemoryFission.

**Expected speedup.** ~2.2x (dominated by fission; the cross-fn
component is marginal in wall-clock).

```
Variants:
  E7a: 2 consumers, writer via global, 100K elements → ~2.2x
  E7b: 4 consumers, writer via global, 50K elements  → ~3.0x
```

---

## Extension E8: Partial Remat — Lookup Table with Expensive Index Computation

**Pattern.** Producer: `buf[i] = lut[hash(i) % K]` where `hash` is a
cheap integer hash (xor-shift). Consumer reads `buf[i]`. The leaf
load `lut[...]` is from a small buffer, but the index computation
involves non-trivial arithmetic that must also be rematerialized.

**Why it matters.** In real workloads, the index into the leaf buffer
often involves non-trivial computation (modular arithmetic, bit
manipulation). This tests whether the cost model correctly accounts
for index-computation ALU cost in the partial-remat decision.

**Strategy.** Partial remat (S2b).

**Expected speedup.** 1.2–1.5x (more ALU work per element than the
simple `i % K` pattern).

```
Variants:
  E8a: xor-shift hash, lut 32, buf 16MB → ~1.3x
  E8b: multiply-shift hash, lut 64, buf 16MB → ~1.4x
```

---

## Extension E9: Partial Remat — Multiple Consumer Loops

**Pattern.** Producer fills a 16MB buffer. Three consumer loops each
read the entire buffer with different reduction operations (sum, max,
dot-product with a second buffer). Partial remat fires on all three
consumers independently.

**Why it matters.** Tests whether partial remat correctly handles
multiple independent consumer loops targeting the same producer buffer.
Each consumer gets its own cloned computation tree.

**Strategy.** Partial remat (S2b), fires 3 times.

**Expected speedup.** 1.5–2.5x (each consumer's DRAM load is replaced;
3 consumers amplify the total bandwidth saving).

```
Variants:
  E9a: 2 consumers, 16MB buf → ~1.5x
  E9b: 3 consumers, 16MB buf → ~2.0x
  E9c: 4 consumers, 64MB buf → ~2.5x
```

---

## Extension E10: Fission at Extreme Consumer Count

**Pattern.** 8 and 16 sibling loops with identical `sqrt+div` chain.
Tests whether fission speedup continues to scale.

**Strategy.** MemoryFission.

**Expected speedup.** Following the trend: 8→~5x, 16→~8x (approaching
N/2 for N consumers, limited by the producer+buffer overhead).

```
Variants:
  E10a: 8 consumers, sqrt+div, 25K elements → ~4-5x
  E10b: 16 consumers, sqrt+div, 10K elements → ~6-8x
```

---

## Priority and Dependencies

| Ext | Requires pass fix? | Confidence | Speedup range |
|-----|-------------------|------------|---------------|
| E1 (strided remat) | No | High | 1.5–3x |
| E9 (multi-consumer remat) | No | High | 1.5–2.5x |
| E10 (fission 8/16 consumers) | No | High | 4–8x |
| E3 (2D partial remat) | Maybe (2D affine) | Medium | 1.4–2x |
| E4 (F.1 extraction) | No | Medium | 1.05–1.3x |
| E7 (combined pipeline) | No | Medium | ~2.2x |
| E8 (expensive index) | No | Medium | 1.2–1.5x |
| E2 (multi-leaf) | Yes (dominance fix) | Low | 1.3–1.8x |
| E5 (chained depth-2) | Maybe (weak linkage) | Low | 1.05–1.15x |
| E6 (buffer elim) | Needs testing | Low | 1.10–1.20x |

**Recommended implementation order:** E1, E9, E10, E3, E8, E4, E7.
Skip E2 until the dominance bug is fixed. E5 and E6 have low expected
wall-clock impact and should be deprioritized.

---

## Known Pass Limitations Discovered During Benchmarking

1. **Dominance violation on multi-leaf partial remat.** When the
   producer uses `affine.apply affine_map<(d0) -> (d0 mod K)>` and
   the remat kernel clones this at the consumer, the cloned apply's
   operand (the producer IV) doesn't dominate the consumer site.
   Observed on `pr_moderate_alu` and `pr_3leaves` exploration patterns.

2. **Cross-function scalar forwarding is runtime-neutral.** DR
   eliminates the global load, but the global is L1-hot (8 bytes,
   permanently cached). The saved 4-cycle L1 load is offset by the
   extra computation cloned at the caller.

3. **Buffer elimination doesn't fire on alloca.** LLVM's SROA handles
   alloca before DR gets a chance. For buffer-elim to matter at runtime,
   the buffer must be heap-allocated (`memref.alloc` → `malloc`).

4. **Partial remat cost gate rejects L2-sized buffers.** For a 256KB
   buffer (L2 latency = 12cy), `alu + leafCost >= loadLatency` fails
   because the margin is too thin. Partial remat only provides
   measurable wins when the consumer buffer is ≥1MB (L3/DRAM latency).

5. **Fission speedup plateaus near N/2.** With N consumer loops, the
   theoretical maximum is Nx (eliminate N-1 redundant computes). In
   practice, the producer loop + buffer overhead caps the speedup at
   ~N/2 for N≥4.
