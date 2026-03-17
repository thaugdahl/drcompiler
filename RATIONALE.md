# Data Recomputation: Rationale

## Core Insight

Polygeist/cgeist at `-O2` greedily fuses producer-consumer loop pairs,
eliminating intermediate buffers by recomputing values inline. This is the
right decision in many cases — but not all.

When a buffer is **small enough to fit in L1/L2** and its computation is
**expensive** (sqrt, division, transcendentals), keeping the buffer and
reading from it is cheaper than recomputing. Conversely, when a buffer is
**large** (exceeds cache capacity) and the computation is **cheap**
(arithmetic on values already in registers), recomputation avoids costly
cache misses.

cgeist has no cost model — it always fuses when it can. The
DataRecomputation pass makes **cache-aware materialization decisions**:
recompute or keep the buffer, based on static analysis of working set size,
computation cost, consumer count, and cache hierarchy parameters.

## Motivating Example

```c
// v[] is 16 KB (fits in L1).  sqrt+div costs ~30 cycles per element.
// An L1 load costs ~4 cycles.  Keeping the buffer saves 2×(30-4)×N cycles.

void compute_values(double *v, const double *x, int n) {
    for (int i = 0; i < n; i++)
        v[i] = sqrt(x[i] * x[i] + 1.0) / (x[i] + 0.001);
}

double sum_values(const double *v, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++)
        s += v[i];
    return s;
}

double max_value(const double *v, int n) {
    double m = v[0];
    for (int i = 1; i < n; i++)
        if (v[i] > m) m = v[i];
    return m;
}

double run(const double *x, int n) {
    double v[2048];             // 16 KB — fits in L1
    compute_values(v, x, n);
    double s = sum_values(v, n);  // consumer 1
    double m = max_value(v, n);   // consumer 2
    return s + m;
}
```

cgeist `-O2` eliminates `v[]` and duplicates the `sqrt + div` computation in
both consumer loops — 2N expensive operations instead of N expensive
operations + 2N cheap L1 loads.

A cache-aware pass would keep the buffer, saving roughly
`N × (cost_sqrt + cost_div - cost_L1_load)` cycles.

## When Recomputation Wins vs. Loses

| Scenario | Buffer size vs. cache | Computation cost | Consumers | Decision |
|---|---|---|---|---|
| Cheap producer, large buffer | > L2 | Low (add, mul) | 1 | **Recompute** — avoid cache miss |
| Cheap producer, small buffer | < L1 | Low | 1 | Either — roughly equal |
| Expensive producer, small buffer | < L1 | High (sqrt, div) | ≥ 2 | **Keep buffer** — L1 hit is cheaper |
| Expensive producer, large buffer | > L2 | High | 1 | Depends on miss penalty vs. ALU cost |
| Any producer, intervening eviction | Evicted by other work | Any | 1 | **Recompute** if cost < miss penalty |

## Pipeline

```
cgeist -O0          Preserves program structure: buffers, calls, loops
    ↓
dr-opt              Cache-aware materialization decisions:
                    - Classify each buffer (size, computation cost, consumers)
                    - Decide: recompute (fuse) or keep (preserve buffer)
                    - For "recompute": fuse producer into consumer, eliminate buffer
                    - For "keep": leave as-is (or introduce buffer if cgeist fused)
    ↓
mlir-opt            Standard lowering (affine → scf → cf → llvm)
    ↓
mlir-translate      MLIR LLVM dialect → LLVM IR
    ↓
clang               Backend codegen
```

The key design choice is running cgeist at `-O0` to preserve the original
program structure, letting dr-opt make informed decisions rather than
accepting cgeist's greedy fusion.

## Cost Model Parameters

The static cache model needs:

1. **Cache hierarchy**: L1/L2/L3 sizes and latencies (from DLTI or target
   description, or pass options)
2. **Buffer size**: Statically known from `memref.alloca`/`memref.alloc` type
   or from malloc argument analysis (raise-malloc-to-memref)
3. **Computation cost**: Estimated by walking the operand tree of the stored
   value — count operations weighted by type (add=1, mul=1, div=10, sqrt=15, etc.)
4. **Consumer count**: Number of distinct load sites (or call sites passing the
   buffer) that read from the buffer
5. **Intervening working set**: Memory touched between the last store and each
   load — if this exceeds cache capacity, the buffer is likely evicted

## Current Implementation

### Pass Options (dr-opt)

```
--dr-recompute        Enable the transformation (rematerialization)
--dr-cost-model       Gate recomputation on the cache-aware cost model
--dr-l1-size=N        L1 data cache size in bytes (default: 32768)
--dr-l2-size=N        L2 cache size in bytes (default: 262144)
--dr-l1-latency=N     L1 hit latency in cycles (default: 4)
--dr-l2-latency=N     L2 hit latency in cycles (default: 12)
--dr-l3-latency=N     L3 hit latency in cycles (default: 40)
--dr-mem-latency=N    DRAM latency in cycles (default: 200)
```

### Cost Model Decision

For each allocation root with SINGLE-provenance loads:

1. **Estimate buffer size** from `memref.alloc` type
2. **Estimate compute cost** by walking the operand tree with weighted ops:
   - add/sub/cmp/shift/logic: 1 cycle
   - mul: 3 cycles
   - div/rem: 15 cycles
   - sqrt/exp/log/sin/cos/tanh/pow: 20 cycles
3. **Estimate load latency** from buffer size vs cache: L1 hit (4), L2 hit
   (12), L3 hit (40)
4. **Compare**: `recompute_cost = consumers × compute_cost` vs
   `keep_cost = compute_cost + 1 + consumers × load_latency`
5. If keep is cheaper, skip rematerialization for this buffer

### Example Decisions

| Buffer | Size | Compute | Load | Consumers | Keep Cost | Recompute Cost | Decision |
|--------|------|---------|------|-----------|-----------|----------------|----------|
| `memref<1xf64>` (sqrt+div) | 8B | 40 | 4 | 2 | 49 | 80 | **KEEP** |
| `memref<1xi32>` (add) | 4B | 1 | 4 | 1 | 6 | 1 | **RECOMPUTE** |

### Future Work

- **Intervening working set estimation**: Walk ops between store and load,
  sum sizes of all memrefs accessed in between.  If this exceeds cache
  capacity, increase estimated load latency (buffer likely evicted).
- **Loop-carried analysis**: When store and load are in different iterations
  or different loops, model the full working set of the loop body.
- **Producer-consumer loop fusion**: When the cost model says "recompute",
  actually fuse the producer loop into the consumer loop (currently only
  handles scalar rematerialization).
- **Evaluation on SPEC CPU 2017**: Compare `cgeist -O2` (greedy fusion) vs
  `cgeist -O0 → dr-opt --dr-cost-model` (cache-aware decisions).
