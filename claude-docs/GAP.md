# Gap Analysis: Memory Fission Pass

## The Optimization Problem

Polyhedral optimizers (e.g., cgeist -O2, Polly, PLUTO) aggressively fuse
producer-consumer loop nests to improve data locality. However, when multiple
**sibling loops** independently consume the same input array and each
recomputes the same expensive expression (sqrt, div, transcendentals), no
existing pass in the standard MLIR or LLVM pipeline eliminates the redundancy.

### Concrete pattern

```c
// Three sibling loops, each computing sqrt(x[i])/scale independently.
// The expensive chain (cost ~35 ALU cycles) is duplicated 3x.
for (int i = 0; i < n; i++)
    a[i] += sqrt(x[i]) / scale + 1.0;
for (int i = 0; i < n; i++)
    b[i] += sqrt(x[i]) / scale * 2.5;
for (int i = 0; i < n; i++)
    c[i] += sqrt(x[i]) / scale - 0.5;
```

Memory Fission materializes the shared computation into a buffer:

```c
// Producer: compute once, store to buffer
for (int i = 0; i < n; i++)
    buf[i] = sqrt(x[i]) / scale;
// Consumers: load from buffer (L1 hit, ~4 cycles) instead of recomputing (~35 cycles)
for (int i = 0; i < n; i++)
    a[i] += buf[i] + 1.0;
for (int i = 0; i < n; i++)
    b[i] += buf[i] * 2.5;
for (int i = 0; i < n; i++)
    c[i] += buf[i] - 0.5;
```

The cost model decides per-candidate:
- **Recompute cost** = num_consumers x compute_cost
- **Keep cost** = compute_cost + store + num_consumers x load_latency
- Fission fires when keep_cost < recompute_cost.

---

## Existing Passes Tested

All tests performed on the `sqrt_consumers` benchmark (3 sibling loops,
sqrt+div chain, N=4000 so buffer fits in L1).

### MLIR-level passes

| Pass | What it does | Why it doesn't help |
|------|-------------|---------------------|
| `--cse` | Common subexpression elimination within a block/region | sqrt ops are in **separate loop bodies** (separate MLIR regions). CSE cannot cross region boundaries. |
| `--affine-loop-fusion` | Fuses producer-consumer loop pairs to eliminate intermediate buffers | These are **sibling loops**, not producer-consumer. No loop writes a buffer that another reads. Fusion legality analysis finds no candidate pairs. |
| `--affine-loop-fusion=maximal` | Same, with aggressive heuristics | Same result — no producer-consumer edge exists between sibling loops. |
| `--canonicalize` | Algebraic simplifications, dead code elimination | No algebraic identity relates the three loop bodies. |

### LLVM-level passes (clang backend)

| Pass / Flag | What it does | Why it doesn't help |
|-------------|-------------|---------------------|
| clang `-O2` | Standard optimization pipeline (GVN, LICM, SLP, loop opts) | GVN/CSE operate on the dominator tree. The three loop bodies are in **separate basic blocks** with no dominance relationship — each is only reachable from its own loop header. |
| clang `-O3` | Adds aggressive vectorization, unrolling | Same sqrt count as -O2. The extra vectorization doesn't help because the redundancy is **across** loops, not within a single loop. |
| `-mllvm -enable-gvn-hoist` | Hoists identical computations from different branches to their common dominator | Only works on **branching** control flow (if/else). Sequential loops are not branches — they execute unconditionally in sequence. |
| LLVM LoopFuse | Fuses adjacent loops with compatible trip counts | Not available / does not fire on this pattern in LLVM 22. Even if it did, it would need CSE within the fused body as a follow-up — the fusion itself doesn't eliminate the redundancy. |

### Summary of sqrt instruction counts (baseline, no fission)

```
                        kernel()    main() (inlined)
MLIR (pre-lowering)     3 sqrt      3 sqrt
LLVM IR                 3 sqrt      3 sqrt
Assembly (clang -O2)    3 sqrtpd    3 sqrtpd
Assembly (clang -O3)    3 sqrtpd    3 sqrtpd
+ GVN-hoist             3 sqrtpd    3 sqrtpd
+ MLIR CSE              3 sqrt      3 sqrt
+ affine-loop-fusion    3 sqrt      3 sqrt
```

After Memory Fission: **1 sqrtpd** in the producer loop per function.

---

## Why the Gap Exists

The fundamental issue is a **scope mismatch**: all existing redundancy
elimination passes (CSE, GVN, PRE, GVN-hoist) operate within a single
region, basic block, or dominator subtree. Sibling loops create **disjoint
scopes** — the computation in loop 2 is neither dominated by nor in the
same block as the computation in loop 1.

Loop fusion could in principle merge the siblings into one loop (enabling
intra-body CSE), but:

1. **MLIR's `affine-loop-fusion`** only considers producer-consumer pairs
   (loops connected by a read-after-write through a buffer). Sibling loops
   sharing a read-only input have no such edge.

2. **LLVM's LoopFuse** targets adjacent loops with compatible headers but
   does not fire on the MLIR-lowered IR, likely due to the memref-to-pointer
   lowering obscuring the loop structure.

3. Even if fusion succeeded, it would need a **follow-up CSE** pass to
   actually eliminate the redundancy. The fusion itself just co-locates
   the computations; a separate pass must recognize they're identical.

Memory Fission takes the opposite approach: instead of fusing loops
(eliminating a buffer), it **creates** a buffer (allowing loops to stay
separate). This is correct by construction — the producer loop computes
exactly what each consumer needs — without requiring post-fusion CSE.

---

## Benchmark Results

Compiled with: `cgeist -O2 → dr-opt → mlir-opt (lower) → clang -O2`

| Benchmark | Consumers | Compute cost | Baseline | Fission | Speedup |
|-----------|-----------|-------------|----------|---------|---------|
| div_chain | 4 | 35 (sqrt+div) | 1.234s | 0.429s | **2.88x** |
| sqrt_consumers | 3 | 35 (sqrt+div) | 1.555s | 0.667s | **2.33x** |
| transcendental_chain | 2 | 40 (exp+log) | 2.905s | 1.609s | **1.81x** |

Speedup scales with consumer count:
- 4 consumers → 2.88x (cost model predicts 140/52 = 2.69x)
- 3 consumers → 2.33x (cost model predicts 105/48 = 2.19x)
- 2 consumers → 1.81x (cost model predicts 80/49 = 1.63x)

N=4000 (32KB buffer) ensures the materialized buffer fits in L1 cache.

---

## Prior Work

### Closest existing work: Halide's compute/store tradeoff

- **Ragan-Kelley et al., "Halide: A Language and Compiler for Optimizing
  Parallelism, Locality, and Recomputation in Image Processing Pipelines"
  (PLDI 2013).** Halide's scheduling model explicitly parameterizes
  recompute-vs-materialize: a `Func` can be `compute_inline` (recompute at
  every consumer — no buffer), `compute_root` (materialize into a buffer
  once, all consumers read from it), or `compute_at` (materialize at a
  specific loop level). This is the same fundamental tradeoff Memory
  Fission addresses.

- **Adams et al., "Learning to Optimize Halide with Tree Search and Random
  Programs" (SIGGRAPH 2019).** Halide's auto-scheduler uses a cost model
  balancing arithmetic cost against memory traffic to choose between
  compute_inline and compute_root — directly analogous to our
  recompute_cost vs. keep_cost decision.

**Key difference:** Halide operates on a dataflow graph of image processing
stages (Funcs), not on general-purpose loop nests. The schedule is an
explicit user- or auto-scheduler annotation on a DSL. Memory Fission
operates on opaque MLIR affine loop nests produced by a general-purpose C
compiler (cgeist), detecting the materialization opportunity via structural
fingerprinting without any programmer annotation.

### Loop fusion/fission cost models

- **Kennedy & McKinley, "Maximizing Loop Parallelism and Improving Data
  Locality via Loop Fusion and Distribution" (1993).** Foundational typed
  fusion framework. Fission (distribution) is used to enable parallelism
  or fix dependences, not to eliminate redundant computation. The cost
  model considers parallelism and reuse, but not cross-loop computation
  duplication.

- **Kennedy & McKinley, "Typed Fusion with Applications to Parallel and
  Sequential Code Generation" (1994).** Formalizes when fusion is
  profitable vs. when fission is better, considering register pressure
  and reuse. The typed fusion framework explicitly models keeping values
  in registers (fusion) vs. materializing to memory (fission), but does
  not model the cost of redundant computation across unfused loops.

- **Megiddo & Sarkar, "Optimal Weighted Loop Fusion for Parallel Programs"
  (SPAA 1997).** Models fusion as a weighted graph problem balancing
  parallelism and locality. Does not consider the case where unfused loops
  redundantly recompute expensive expressions.

- **Darte, "On the Complexity of Loop Fusion" (PACT 2000).** Formalizes
  the NP-completeness of optimal loop fusion. The inverse problem (optimal
  fission to enable materialization) inherits this complexity.

- **McKinley, Carr & Tseng, "Improving Data Locality with Loop
  Transformations" (TOPLAS 1996).** Comprehensive treatment of how loop
  transformations affect cache behavior. Their cost model considers cache
  line reuse across loop nests but not per-element computation cost.

### Redundancy elimination across scopes

- **Morel & Renvoise, "Global Optimization by Suppression of Partial
  Redundancies" (CACM 1979).** PRE eliminates computations that are
  redundant on some but not all paths. This is the theoretical basis for
  GVN-PRE in LLVM. It operates on the CFG dominator tree — separate loop
  bodies are disjoint paths with no dominator containing the redundant
  computation.

- **Knoop, Ruthing & Steffen, "Lazy Code Motion" (PLDI 1992).** Extends
  PRE with optimal placement. Same limitation: the analysis scope is the
  CFG, not across separate loop nests.

### Polyhedral frameworks

- **Bondhugula et al., "A Practical Automatic Polyhedral Parallelizer and
  Locality Optimizer" (PLDI 2008) — PLUTO.** Polyhedral fusion uses
  affine dependence analysis. Sibling loops with no data dependence can
  be fused (tiled together), but the framework has no mechanism to detect
  or exploit shared computation across the fused/unfused loops. The tiling
  cost model considers reuse distances, not computation cost.

- **Verdoolaege et al., "Polyhedral Parallel Code Generation for CUDA"
  (TACO 2013) — PPCG.** Separates computation into tiles and materializes
  shared data into GPU shared memory via "copy" loops. Structurally
  similar to Memory Fission (producer loop + buffer + consumer loops), but
  materializes raw *data* (array tiles), not *computed expressions*. No
  computation deduplication.

- **Polly (LLVM).** Uses ISL for polyhedral analysis. Can perform maximal
  fusion and tiling but does not analyze or eliminate redundant computation
  across the fused/distributed loops.

- **MLIR `affine-loop-fusion` (upstream).** Implements producer-consumer
  fusion following Bondhugula's dependence-based approach. Only considers
  loops connected by a read-after-write edge. Sibling loops sharing a
  read-only input are invisible to the analysis.

- **Polygeist / cgeist (Moses et al., 2021).** Raises C to MLIR affine
  dialect. At -O2 it runs polyhedral fusion, which can create the
  sibling-loop-with-duplicated-computation pattern. No built-in mechanism
  to detect or address the resulting redundancy.

### What's new in Memory Fission

No prior work combines all three of:

1. **Cross-loop-nest redundancy detection** via structural fingerprinting:
   identifies identical expensive computation subtrees across sibling loops
   that share a read-only source memref, without programmer annotation.

2. **Cache-aware cost model** for the materialize-vs-recompute decision:
   weighs ALU cost of recomputation against cache hierarchy latencies for
   the materialized buffer.

3. **Operates on general-purpose MLIR affine loop nests**, filling a gap
   between Halide's DSL-level scheduling (which requires a dataflow graph)
   and LLVM's scalar optimizations (which cannot cross loop boundaries).

Halide solves the same recompute-vs-materialize tradeoff but requires
a domain-specific dataflow representation. Memory Fission recovers the
opportunity from opaque, compiler-generated loop nests via fingerprinting.

---

## Recommended citations

For a related work section, the key references are:
- **Ragan-Kelley et al. (2013), Adams et al. (2019)** — Halide (closest)
- **Kennedy & McKinley (1993, 1994)** — fusion/fission cost models
- **Bondhugula et al. (2008)** — PLUTO, polyhedral fusion
- **Verdoolaege et al. (2013)** — PPCG shared memory materialization
- **Morel & Renvoise (1979), Knoop et al. (1992)** — PRE scope limitations
- **Allen & Kennedy (2001)** — textbook: loop distribution
- **Darte (2000)** — complexity of optimal fusion

Note: citations should be verified against the actual publications.
Web search was unavailable during this analysis.
