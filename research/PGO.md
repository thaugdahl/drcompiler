# Profile-Guided Cost Model for Memory Fission

The static cost model in `MemoryFission.cpp` uses hardcoded ALU cycle estimates
(e.g. `sqrt` = 20, `divf` = 15) and assumes L1 load latency for buffer access.
These numbers are architecture-dependent approximations. This document outlines
strategies for replacing them with measured costs, ranging from a one-time
machine calibration to full program-level PGO.

## 1. Machine Calibration (Microbenchmark Suite)

### Overview

A standalone benchmark program that measures the actual latency of each
operation type and memory hierarchy level on the target machine. Run once per
target, emit a cost table, and load it into the pass at compile time. This
replaces the hardcoded constants with measured values — no instrumentation of
the user's program required.

### What to Measure

**ALU operation latencies.** For each op type in the cost model (`addf`,
`mulf`, `divf`, `sqrt`, `exp`, `sin`, `cos`, `tanh`, `log`, `powf`, casts,
etc.), measure the **latency** (not throughput) in cycles. Latency matters
because the ops in a computation chain form a serial dependency — each op
waits for the previous one's result.

Benchmark structure per op:

```c
// Serial dependency chain — measures latency.
// N iterations, result of each feeds into the next.
double x = 1.0001;
uint64_t t0 = __rdtsc();
for (int i = 0; i < N; i++)
    x = sqrt(x);          // data dependency forces serial execution
uint64_t t1 = __rdtsc();
double latency = (double)(t1 - t0) / N;
```

The input value must be chosen to avoid special-case fast paths (e.g.
`sqrt(0)`, `exp(0)`) and NaN/Inf propagation. Use a value like `1.0001`
and vary it to confirm stability.

**Important: prevent optimization.** The compiler will try to eliminate the
loop or hoist the computation. Options:
- Use `volatile` or inline asm barriers (`asm volatile("" : "+x"(x))`)
- Compile the benchmark at `-O1` with the loop marked `nounroll`
- Use `__attribute__((noinline))` on the measured function

**Memory hierarchy latencies.** Use a pointer-chasing benchmark at different
working set sizes to measure load latency at each cache level:

```c
// Pointer chase through an array of shuffled indices.
// Working set size determines which cache level is exercised.
size_t idx = 0;
uint64_t t0 = __rdtsc();
for (int i = 0; i < N; i++)
    idx = array[idx];     // serial load dependency
uint64_t t1 = __rdtsc();
```

| Working set size | Expected level |
|------------------|----------------|
| 8-32 KB          | L1             |
| 128-256 KB       | L2             |
| 2-8 MB           | L3             |
| > LLC size       | DRAM           |

This gives measured values for `l1Latency`, `l2Latency`, etc. that the pass
can use when estimating the cost of buffer loads.

### Output Format

A JSON file emitted by the benchmark, loaded by the pass:

```json
{
  "target": "x86_64-zen4",
  "alu_latency_cycles": {
    "arith.addf": 3,
    "arith.mulf": 4,
    "arith.divf": 13,
    "math.sqrt":  13,
    "math.exp":   18,
    "math.sin":   55,
    "math.cos":   55,
    "math.tanh":  40,
    "math.log":   19,
    "math.powf":  65,
    "arith.sitofp": 3,
    "arith.addi": 1,
    "arith.muli": 3,
    "arith.divsi": 14
  },
  "load_latency_cycles": {
    "l1": 4,
    "l2": 12,
    "l3": 40,
    "dram": 200
  }
}
```

### Integration with the Pass

At the start of `runOnOperation()`, check for a calibration file (path
provided via a pass option, e.g. `--cost-table=<path>`). If present, use
it to populate `opCost()` lookups and the load latency parameter. If absent,
fall back to the current hardcoded values.

`opCost()` becomes a table lookup keyed on `op->getName().getStringRef()`
instead of an `isa<>` cascade.

### Limitations

- Measures **isolated** op latency. In practice, out-of-order execution
  overlaps independent ops, so a chain of `addf + mulf` may cost less than
  `latency(addf) + latency(mulf)`. The calibration overpredicts chain cost
  slightly, which biases toward fission (buffer more aggressively). This is
  conservative and generally acceptable.
- Does not capture cache-level selection for the buffer. The benchmark tells
  you L1 costs 4 cycles and L3 costs 40, but not which level a specific
  buffer will land in at runtime. That depends on trip count and surrounding
  memory pressure — only profiling (sections 2-3) can capture that.
- Transcendental latencies vary by input range on some microarchitectures
  (e.g. Intel's `fsin` microcode). The benchmark should sample across a
  representative input range and report the median.

## 2. Instrumentation PGO (Chain-Level Profiling)

### Overview

Two-pass compilation. The first pass instruments each fission candidate to
measure its actual per-invocation cost. The second pass loads the profile and
uses measured costs in the fission decision.

### Instrumentation Pass

For each computation chain identified by `extractChains()`, emit cycle-counter
probes around the chain ops inside the loop body:

```
%t0 = call @__drcomp_rdtsc()
// ... chain ops (sqrt, divf, etc.) ...
%t1 = call @__drcomp_rdtsc()
call @__drcomp_log_chain(%fingerprint_id, %t1 - %t0)
```

At program exit, the runtime dumps a profile mapping fingerprint IDs to
average cycles per invocation.

### Key Design Decisions

**Keying on fingerprint.** The chain's structural fingerprint (already computed
by the pass) is a natural profile key. It is stable across compilation runs as
long as the computation structure is unchanged, so a profile collected once can
be reused.

**Amortizing probe overhead.** `rdtsc` costs ~20 cycles on x86. For short
chains (cost ~15-30 cycles) this is significant. Options:
- Sample every Nth iteration instead of every iteration.
- Probe at the loop level (total loop time / trip count) rather than per-chain.
- Accept the overhead for profiling runs — it only needs to be directionally
  correct, not cycle-accurate.

**Profiling the memory side.** The static model assumes L1 latency for buffer
loads (`l1Latency` parameter). This is wrong when the buffer exceeds L1.
To profile this, instrument the *buffered* version too:
- Emit `rdtsc` around the `affine.load` from the buffer in each consumer loop.
- This gives measured load latency under realistic access patterns.
- The decision becomes: `measured_chain_cost * (N - 1)` vs
  `measured_load_cost * N`, where N is the number of consumers.

### Profile Format

A simple text/binary file mapping fingerprint hashes to measured costs:

```
# fingerprint_hash  chain_cycles  load_cycles  trip_count
0xA3F1...           45            4             1024
0xB7E2...           112           7             2048
```

### Optimize Pass

Load the profile at the start of `MemoryFissionPass::runOnOperation()`.
When evaluating a `FissionCandidate`:
1. Look up the candidate's fingerprint in the profile.
2. If found, use `measured_chain_cycles` and `measured_load_cycles` instead of
   the static `opCost()` sum and `l1Latency`.
3. If not found (cold chain), fall back to the static cost model.

## 3. A/B Profiling (Direct Comparison)

### Overview

Instead of modeling the cost tradeoff, emit **both** the buffered and
recomputed versions, profile each, and select the winner.

### Mechanism

For each fission candidate, the instrumentation pass emits:

```c
if (__drcomp_variant_select(fingerprint_id)) {
    // Path A: original (recomputed) loops
    for (i = 0..n) { ... sqrt(A[i]) ... }  // loop 1
    for (i = 0..n) { ... sqrt(A[i]) ... }  // loop 2
} else {
    // Path B: fissioned (buffered) version
    for (i = 0..n) { buf[i] = sqrt(A[i]); }
    for (i = 0..n) { ... buf[i] ... }       // loop 1
    for (i = 0..n) { ... buf[i] ... }       // loop 2
}
```

The runtime alternates between paths (or runs each for a fixed number of
iterations) and records aggregate cycle counts per path per fingerprint.

### Profile Format

```
# fingerprint_hash  recompute_cycles  buffered_cycles  winner
0xA3F1...           89000             52000            buffered
0xB7E2...           45000             61000            recompute
```

### Optimize Pass

Load the profile. For each candidate, if the profile says `buffered` wins,
apply the fission transform. Otherwise, skip. No cost model involved — pure
empirical selection.

### Tradeoffs

| Aspect                  | Calibration (1)           | Instrumentation PGO (2)  | A/B Profiling (3)        |
|-------------------------|---------------------------|--------------------------|--------------------------|
| Profiled artifact       | Target machine            | User program             | User program             |
| Runs per target         | Once                      | Once per program/input   | Once per program/input   |
| Profile accuracy        | Per-op (isolated)         | Per-chain (in context)   | End-to-end               |
| Captures cache effects  | No (provides table only)  | Only if load is profiled | Automatically            |
| Captures ILP/scheduling | No                        | No                       | Yes                      |
| Code size during prof.  | Separate binary           | ~1x (probes only)        | ~2x (both variants)      |
| Generalizability        | Any program on same arch  | Transfers across inputs  | Tied to exact input size |
| Implementation effort   | Low                       | Moderate                 | Higher (dual codegen)    |

### Recommendation

The three approaches are complementary and layer naturally:

1. **Calibration** as the baseline — replace hardcoded constants with measured
   per-op latencies. Cheap to implement, benefits all programs on the target
   machine, and already a significant improvement over guessed constants.
2. **Instrumentation PGO** for programs where calibration makes borderline
   decisions — the per-chain and per-load measurements resolve cases where
   ILP or cache behavior makes the static model inaccurate.
3. **A/B profiling** as a validation/oracle tool — confirms whether the cost
   model (calibrated or PGO-informed) is making the right call. For SPEC CPU
   2017 with a fixed benchmark set and target machine, it can also serve as
   the production strategy directly.