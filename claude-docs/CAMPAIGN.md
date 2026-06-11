# Synthetic Benchmark Campaign

**Goal:** 200 MLIR programs (including scaled variants) that systematically expose
strengths, weaknesses, and ambiguities of DataRecomputation, MemoryFission, and
the CacheCostModel. Every program is self-contained `.mlir` with lit `// RUN:`
lines so the full suite runs under `llvm-lit`.

**Non-goals:** Real-world representativeness. These are *micro-kernels* that
isolate specific cost model decisions, analysis corner cases, and strategy
interactions. Real workloads (SPEC, ONNX models) remain the validation layer.

---

## Taxonomy

| Category | ID range | Count | Primary target |
|----------|----------|-------|----------------|
| A. Classification stress | A001–A030 | 30 | Provenance analysis accuracy |
| B. Strategy selection | B001–B040 | 40 | Recomputation strategy coverage |
| C. Cost model boundaries | C001–C040 | 40 | Decision boundary sensitivity |
| D. Memory fission | D001–D030 | 30 | Fission profitability & correctness |
| E. Buffer elimination | E001–E020 | 20 | Whole-buffer elim feasibility |
| F. Scaling & stress | F001–F020 | 20 | Compile-time, pathological cases |
| G. Cross-feature interaction | G001–G020 | 20 | Pass composition, ordering effects |
| **Total** | | **200** | |

Each program file is named `bench/<category>/<id>-<short-name>.mlir`.

---

## A. Classification Stress (A001–A030)

Test the four provenance categories (SINGLE/MULTI/LEAKED/KILLED) under
progressively harder conditions.

### A.1 SINGLE provenance — clean cases (A001–A006)

| ID | Name | Description | Scaling |
|----|------|-------------|---------|
| A001 | `single-scalar` | Rank-0 alloc, one store, one load | Baseline |
| A002 | `single-indexed-1d` | 1D memref, affine store + load at same index | Trip count: 64, 256, 1024 |
| A003 | `single-indexed-2d` | 2D memref, nested affine store + load | Dims: 32x32, 128x128 |
| A004 | `single-indexed-3d` | 3D memref, triple-nested | Dims: 8x8x8, 32x32x32 |
| A005 | `single-cross-fn` | Store in callee, load in caller | Call depth: 1, 2, 3 |
| A006 | `single-global` | Store to global memref, load elsewhere | 1 vs 4 globals |

### A.2 MULTI provenance (A007–A012)

| ID | Name | Description | Scaling |
|----|------|-------------|---------|
| A007 | `multi-two-stores` | Two stores to same index, one load | — |
| A008 | `multi-if-else` | Store in if-branch, different store in else-branch | Nesting: 1, 2, 3 levels |
| A009 | `multi-loop-overwrite` | Store in loop body overwrites prior iteration's store | Trip: 4, 64, 1024 |
| A010 | `multi-cross-fn-two-sites` | Same callee called from 2 sites with different stores | — |
| A011 | `multi-view-alias` | Two stores through different views of same base | Subview count: 2, 4 |
| A012 | `multi-conditional-index` | Store at index `i` or `i+1` depending on condition | — |

### A.3 LEAKED provenance (A013–A018)

| ID | Name | Description | Scaling |
|----|------|-------------|---------|
| A013 | `leaked-external-call` | Memref passed to external (declaration-only) function | — |
| A014 | `leaked-return` | Memref returned from function | — |
| A015 | `leaked-ptr-cast` | Memref cast to pointer via memref_to_pointer | — |
| A016 | `leaked-global-clobber` | Call clobbers global memref (nullptr provenance entry) | — |
| A017 | `leaked-nested-call` | Memref escapes through 2-hop call chain to external | Depth: 2, 3 |
| A018 | `leaked-partial` | Some indices SINGLE, others LEAKED (mixed) | — |

### A.4 KILLED provenance (A019–A024)

| ID | Name | Description | Scaling |
|----|------|-------------|---------|
| A019 | `killed-no-store` | Load with no prior store (fresh alloc) | — |
| A020 | `killed-overwrite` | Rank-0 store kills prior rank-0 store; load after second | — |
| A021 | `killed-rank0-indexed` | Rank-0 store kills indexed writes (coarse coverage) | — |
| A022 | `killed-zero-trip` | For loop with possible zero trip, no prior store | Affine + SCF variant |
| A023 | `killed-conditional-store` | Store only in if-branch, load after if (no else) | — |
| A024 | `killed-store-after-load` | Store dominates load in text but load executes first (in loop) | — |

### A.5 Ambiguous / boundary cases (A025–A030)

| ID | Name | Description | Scaling |
|----|------|-------------|---------|
| A025 | `ambig-dynamic-index` | Dynamic (non-affine) store index; conservative may-write | — |
| A026 | `ambig-symbolic-bound` | Symbolic loop bound prevents trip-count reasoning | — |
| A027 | `ambig-aliased-view` | Subview + reinterpret_cast chain obscures base | Chain length: 2, 3, 4 |
| A028 | `ambig-iv-arithmetic` | Index computed via IV subtraction across sibling loops | — |
| A029 | `ambig-mixed-dialect` | Store via LLVM dialect, load via memref dialect (flat coverage) | — |
| A030 | `ambig-phase-aware` | Phase-aware seeding: store reachable only on 2nd iteration | — |

---

## B. Strategy Selection (B001–B040)

Each program is designed so exactly one recomputation strategy is optimal or
applicable, exposing strategy coverage and selection priority.

### B.1 DirectForward (B001–B006)

| ID | Name | Description | Scaling |
|----|------|-------------|---------|
| B001 | `direct-scalar-const` | Store arith.constant, load same block | — |
| B002 | `direct-scalar-arg` | Store function arg, load same block | — |
| B003 | `direct-cross-block` | Store in entry, load in successor block (SSA dominance) | Block depth: 2, 4 |
| B004 | `direct-loop-invariant` | Store loop-invariant value, load inside loop | Trip: 64, 1024 |
| B005 | `direct-multi-consumer` | One store, 4 loads of same value in different blocks | Consumer count: 2, 4, 8 |
| B006 | `direct-cross-fn-arg` | Callee stores caller's arg; caller loads | — |

### B.2 FullRemat (B007–B016)

| ID | Name | Description | Scaling |
|----|------|-------------|---------|
| B007 | `remat-add-chain` | Store result of 2-op add chain, load elsewhere | Chain: 2, 4, 8 ops |
| B008 | `remat-mul-chain` | Store result of multiply chain | Chain: 2, 4, 8 ops |
| B009 | `remat-mixed-arith` | Store result of add+mul+sub chain | Chain: 3, 6, 12 ops |
| B010 | `remat-with-constant` | Chain includes arith.constant ops (zero cost) | Const fraction: 1/3, 2/3 |
| B011 | `remat-type-convert` | Chain includes sitofp/fptosi conversions | — |
| B012 | `remat-nested-loop` | Remat inside doubly-nested loop | Outer trip: 8, 64 |
| B013 | `remat-diamond-dag` | DAG-shaped computation (shared intermediate) | Fan-out: 2, 4 |
| B014 | `remat-long-chain` | Very long linear chain (16, 32, 64 ops) | Tests compile time |
| B015 | `remat-cross-fn-chain` | Chain spans caller → callee boundary | — |
| B016 | `remat-multi-consumer-chain` | Same chain value consumed by 2, 4 loads | Consumer count scaling |

### B.3 PartialRemat (B017–B026)

| ID | Name | Description | Scaling |
|----|------|-------------|---------|
| B017 | `partial-one-leaf` | Chain with 1 non-chainable leaf load | — |
| B018 | `partial-two-leaves` | Chain with 2 non-chainable leaf loads | — |
| B019 | `partial-max-leaves` | Chain with exactly `dr-partial-max-leaves` (4) leaves | — |
| B020 | `partial-over-budget` | Chain with 5 leaves (exceeds default budget) | Budget: 4, 6, 8 |
| B021 | `partial-strided-leaf` | Leaf load with stride > 1 (cache-line-aware cost) | Stride: 1, 4, 16 |
| B022 | `partial-global-const-leaf` | Leaf load from memref.global constant | — |
| B023 | `partial-writeonce-leaf` | Leaf buffer written exactly once (safe for remat) | — |
| B024 | `partial-intervening-store` | Intervening store between leaf write and remat point | — |
| B025 | `partial-writer-in-region` | Leaf's writer is inside scf.if region (blocked) | — |
| B026 | `partial-escape-blocks` | Leaf buffer escapes via pointer cast (blocked) | — |

### B.4 InterprocRemat (B027–B033)

| ID | Name | Description | Scaling |
|----|------|-------------|---------|
| B027 | `interproc-simple` | Callee stores, caller loads, simple chain | — |
| B028 | `interproc-deep-chain` | Callee stores result of 8-op chain | — |
| B029 | `interproc-two-args` | Callee stores to two different arg buffers | — |
| B030 | `interproc-depth2` | Store in callee's callee; load in top-level caller | — |
| B031 | `interproc-depth3` | Three-hop interprocedural chain | — |
| B032 | `interproc-impure-block` | Callee chain includes call (impurity blocks remat) | — |
| B033 | `interproc-multi-site` | Same callee called 3 times with different arg stores | — |

### B.5 CrossFnOrdered (B034–B037)

| ID | Name | Description | Scaling |
|----|------|-------------|---------|
| B034 | `cross-fn-ordered-simple` | Ordered remat across call boundary | — |
| B035 | `cross-fn-ordered-loop` | Store in callee loop, load in caller loop | Trip: 64, 1024 |
| B036 | `cross-fn-ordered-nested` | Nested loops in callee, flat loop in caller | — |
| B037 | `cross-fn-ordered-conflict` | Two callees store to same buffer (conflict resolution) | — |

### B.6 ConstantGlobalFold (B038–B040)

| ID | Name | Description | Scaling |
|----|------|-------------|---------|
| B038 | `const-global-scalar` | Load from memref.global with constant initializer | — |
| B039 | `const-global-array` | Load from constant global array at known index | Array: 16, 256, 4096 |
| B040 | `const-global-multi-use` | Same global constant loaded in 4 functions | — |

---

## C. Cost Model Boundaries (C001–C040)

Each program is crafted to sit near a decision boundary of the cost model.
Scaled variants sweep across the boundary to expose sensitivity.

### C.1 Buffer size sweeps — cache hierarchy boundaries (C001–C010)

For each: one store, one load, fixed ALU cost. Sweep buffer size across L1/L2/L3
boundaries to verify latency estimate changes.

| ID | Name | Buffer sizes | ALU cost | Expected |
|----|------|-------------|----------|----------|
| C001 | `sweep-l1-fit` | 4KB, 16KB, 30KB | 8 | Recompute (cheap ALU, L1-resident load) |
| C002 | `sweep-l1-boundary` | 30KB, 32KB, 34KB | 8 | Flip at 32KB: L1→L2 latency jump |
| C003 | `sweep-l2-fit` | 64KB, 128KB, 240KB | 8 | Keep (L2 latency, moderate) |
| C004 | `sweep-l2-boundary` | 240KB, 256KB, 280KB | 8 | Flip at 256KB: L2→L3 latency jump |
| C005 | `sweep-l3-boundary` | L3-2KB, L3, L3+2KB | 8 | Flip at L3 size: L3→DRAM |
| C006 | `sweep-dram` | 64MB, 256MB | 8 | Keep (DRAM latency, remat always wins) |
| C007 | `sweep-l1-expensive` | 16KB, 32KB, 48KB | 40 | Keep (expensive ALU, even if L1) |
| C008 | `sweep-l2-cheap` | 64KB, 256KB | 2 | Recompute (trivial ALU, any cache level) |
| C009 | `sweep-unknown-size` | Dynamic alloc size | 8 | Tests fallback heuristic |
| C010 | `sweep-zero-size` | Empty memref (0 elements) | 0 | Degenerate case |

### C.2 ALU cost sweeps — recompute vs. keep threshold (C011–C018)

Fixed buffer size (L2-fit, latency=12). Sweep ALU cost to find break-even.

| ID | Name | ALU cost | Consumers | Expected |
|----|------|----------|-----------|----------|
| C011 | `alu-trivial` | 1 (single add) | 1 | Recompute |
| C012 | `alu-light` | 4 (add chain) | 1 | Recompute |
| C013 | `alu-moderate` | 10 (mixed ops) | 1 | Near boundary |
| C014 | `alu-break-even` | 12 (matches L2 lat) | 1 | Exact tie |
| C015 | `alu-expensive` | 20 (sqrt) | 1 | Keep |
| C016 | `alu-very-expensive` | 40 (div+sqrt) | 1 | Keep |
| C017 | `alu-expensive-many-consumers` | 20 | 4 | Keep amplified by consumers |
| C018 | `alu-trivial-many-consumers` | 2 | 8 | Recompute (amortized) |

### C.3 Consumer count sweeps (C019–C024)

Fixed buffer (128KB, L2-fit) and ALU cost. Sweep consumer count.

| ID | Name | ALU | Consumers | Expected |
|----|------|-----|-----------|----------|
| C019 | `consumers-1` | 8 | 1 | Recompute |
| C020 | `consumers-2` | 8 | 2 | Near boundary |
| C021 | `consumers-4` | 8 | 4 | Keep |
| C022 | `consumers-8` | 8 | 8 | Keep (strongly) |
| C023 | `consumers-1-dram` | 8 | 1 | Recompute (DRAM penalty huge) |
| C024 | `consumers-8-dram` | 8 | 8 | Still recompute? (DRAM so expensive) |

### C.4 Footprint analysis (C025–C030)

Test `dr-footprint-analysis` eviction tracking.

| ID | Name | Description | Expected |
|----|------|-------------|----------|
| C025 | `footprint-no-intervening` | No ops between store and load | Recompute (buffer warm) |
| C026 | `footprint-small-intervening` | 8KB touched between store/load | Recompute (still in L1) |
| C027 | `footprint-l1-evict` | 48KB touched between store/load | Keep (likely evicted from L1) |
| C028 | `footprint-l2-evict` | 512KB touched between store/load | Keep (likely evicted from L2) |
| C029 | `footprint-operand-warm` | Remat operand accessed just before load | Reduced operand penalty |
| C030 | `footprint-operand-cold` | Remat operand not accessed for 1MB | High operand penalty |

### C.5 Stride-aware partial remat cost (C031–C036)

Test cache-line-aware cost for partial remat leaf loads.

| ID | Name | Stride | Elem size | effBytes | Expected |
|----|------|--------|-----------|----------|----------|
| C031 | `stride-unit-f32` | 1 | 4B | 5B | Cheap leaf (~1 cycle fraction) |
| C032 | `stride-unit-f64` | 1 | 8B | 9B | Slightly more expensive |
| C033 | `stride-4-f32` | 4 | 4B | 17B | ~25% of cache line |
| C034 | `stride-16-f32` | 16 | 4B | 64B | Full cache line per access |
| C035 | `stride-irregular` | 7 | 4B | 29B | ~45% of cache line |
| C036 | `stride-unknown` | Dynamic | 4B | — | Fallback to full line cost |

### C.6 Register pressure & code bloat (C037–C040)

Test `dr-reg-budget` and `dr-icache-soft-budget` penalties.

| ID | Name | Description | Expected |
|----|------|-------------|----------|
| C037 | `regpress-under-budget` | Remat chain uses 10 values (under 32 budget) | No spill penalty |
| C038 | `regpress-over-budget` | Remat chain uses 40 values (over 32 budget) | Spill penalty tips to keep |
| C039 | `codebloat-under-budget` | 4 consumers × 20-op chain = 80 ops (under 128) | No bloat penalty |
| C040 | `codebloat-over-budget` | 4 consumers × 40-op chain = 160 ops (over 128) | Bloat penalty tips to keep |

---

## D. Memory Fission (D001–D030)

### D.1 Basic fission (D001–D008)

| ID | Name | Description | Scaling |
|----|------|-------------|---------|
| D001 | `fission-sqrt-2x` | sqrt chain duplicated in 2 loops | — |
| D002 | `fission-div-3x` | div chain in 3 consumer loops | — |
| D003 | `fission-transcendental` | exp/log chain in 2 loops | — |
| D004 | `fission-cheap-chain` | add-only chain (cost < minChainCost=15) | Should NOT fission |
| D005 | `fission-mixed-cost` | sqrt+add chain (mixed expensive/cheap) | — |
| D006 | `fission-many-consumers` | Same chain in 4, 8, 16 consumer loops | Consumer sweep |
| D007 | `fission-single-consumer` | Expensive chain in 1 loop only (no duplication) | Should NOT fission |
| D008 | `fission-identical-fingerprint` | Two structurally identical chains (same ops, diff SSA) | — |

### D.2 Loop structure variants (D009–D016)

| ID | Name | Description | Scaling |
|----|------|-------------|---------|
| D009 | `fission-small-trip` | Affine loop, trip count 8 | — |
| D010 | `fission-medium-trip` | Trip count 256 | — |
| D011 | `fission-large-trip` | Trip count 4096 (buffer exceeds L1) | — |
| D012 | `fission-dynamic-trip` | Dynamic (symbolic) trip count | — |
| D013 | `fission-nested-outer` | Chain in outer loop of 2-deep nest | Outer: 8, 32 |
| D014 | `fission-nested-inner` | Chain in inner loop of 2-deep nest | Inner: 64, 256 |
| D015 | `fission-mismatched-bounds` | Consumer loops have different bounds | Should NOT fission |
| D016 | `fission-non-affine` | SCF.for loop (not affine) | Should skip (affine-only) |

### D.3 Cost model interaction (D017–D024)

| ID | Name | Description | Expected |
|----|------|-------------|----------|
| D017 | `fission-cheap-load` | Materialized buffer fits L1 (latency=4) | Keep (cheap load) |
| D018 | `fission-expensive-load` | Buffer spills to L2 (latency=12) | Depends on chain cost |
| D019 | `fission-dram-spill` | Buffer much larger than L3 | Recompute (DRAM too expensive) |
| D020 | `fission-break-even` | Chain cost ≈ load latency × consumers | Exact boundary |
| D021 | `fission-with-json-cost` | Custom JSON cost model changes op weights | Different decision |
| D022 | `fission-consumer-threshold` | Chain cost between minConsumerCost and minChainCost | Tests both thresholds |
| D023 | `fission-constant-chain` | Chain of all constants (cost=0) | Should NOT fission (free recompute) |
| D024 | `fission-one-expensive-op` | Single div in chain (cost=15 exactly = minChainCost) | Boundary |

### D.4 Source memref constraints (D025–D030)

| ID | Name | Description | Expected |
|----|------|-------------|----------|
| D025 | `fission-readonly-source` | Source memref only read by consumers | OK to fission |
| D026 | `fission-written-source` | Consumer also writes source memref | Blocked |
| D027 | `fission-global-source` | Source is global memref | OK if no writes |
| D028 | `fission-aliased-source` | Source accessed via subview in one consumer | Tests alias tracking |
| D029 | `fission-multi-source` | Chain reads from 2 source memrefs | Both must be safe |
| D030 | `fission-source-escapes` | Source passed to external call between consumers | Blocked? |

---

## E. Buffer Elimination (E001–E020)

### E.1 Full elimination feasible (E001–E006)

| ID | Name | Description | Scaling |
|----|------|-------------|---------|
| E001 | `elim-simple` | Single alloc, 1 store, 1 load, 1 dealloc — all SINGLE | — |
| E002 | `elim-multi-load` | 1 store, 4 loads — all SINGLE | Load count: 2, 4, 8 |
| E003 | `elim-multi-store` | 4 stores to different indices, 4 loads | — |
| E004 | `elim-cross-fn` | Store in callee, loads in caller — all replaceable | — |
| E005 | `elim-with-views` | Access through subview/cast chain — still safe | — |
| E006 | `elim-shared-subexpr` | Multiple loads share CSE-able computation (discount) | — |

### E.2 Elimination blocked (E007–E012)

| ID | Name | Description | Blocking reason |
|----|------|-------------|-----------------|
| E007 | `elim-blocked-escape-call` | Alloc passed to external call | EscapesToCall |
| E008 | `elim-blocked-escape-return` | Alloc returned from function | EscapesViaReturn |
| E009 | `elim-blocked-ptr-cast` | Alloc cast to pointer value | EscapesAsPtrValue |
| E010 | `elim-blocked-multi-load` | Some loads are MULTI (not all replaceable) | Remaining MULTI loads |
| E011 | `elim-blocked-leaked-load` | Some loads are LEAKED | Remaining LEAKED loads |
| E012 | `elim-blocked-unknown-op` | Alloc used by unrecognized op | EscapesUnknown |

### E.3 Cost-gated elimination (E013–E020)

| ID | Name | Description | Expected |
|----|------|-------------|----------|
| E013 | `elim-cost-cheap-remat` | ALU cost low, buffer large | Eliminate (remat cheap) |
| E014 | `elim-cost-expensive-remat` | ALU cost high, buffer small (L1) | Keep (remat expensive) |
| E015 | `elim-cost-regpressure` | Remat requires 40 live values (over budget) | Spill penalty blocks elim |
| E016 | `elim-cost-codebloat` | 8 loads × 20-op chain = 160 ops cloned | Bloat penalty blocks |
| E017 | `elim-cost-many-stores` | 100 stores, 2 loads (store overhead dominates keep-cost) | Eliminate |
| E018 | `elim-cost-capacity` | Buffer 2× L3 (massive capacity penalty) | Eliminate (DRAM penalty huge) |
| E019 | `elim-cost-shared-discount` | 4 loads share identical computation trees | Shared discount helps elim |
| E020 | `elim-drives-strategies` | `dr-buffer-elim-drives-strategies` overrides per-load veto | Tests flag interaction |

---

## F. Scaling & Stress (F001–F020)

Programs designed to test compile-time behavior, analysis precision under scale,
and worst-case scenarios.

### F.1 Chain depth scaling (F001–F004)

| ID | Name | Chain depth | Purpose |
|----|------|-------------|---------|
| F001 | `chain-depth-16` | 16 ops | Baseline remat depth |
| F002 | `chain-depth-64` | 64 ops | Stress SSA walk |
| F003 | `chain-depth-256` | 256 ops | Cost accumulation precision |
| F004 | `chain-depth-1024` | 1024 ops | Compile-time regression test |

### F.2 Buffer count scaling (F005–F008)

| ID | Name | Buffers | Purpose |
|----|------|---------|---------|
| F005 | `buffers-4` | 4 independent allocs | Baseline |
| F006 | `buffers-16` | 16 allocs | Moderate |
| F007 | `buffers-64` | 64 allocs | Stress analysis maps |
| F008 | `buffers-256` | 256 allocs | Worst-case map size |

### F.3 Call graph scaling (F009–F012)

| ID | Name | Functions | Call depth | Purpose |
|----|------|-----------|------------|---------|
| F009 | `callgraph-flat-8` | 8 leaf callees | 1 | Wide call graph |
| F010 | `callgraph-flat-32` | 32 leaf callees | 1 | Stress enriched call graph |
| F011 | `callgraph-deep-4` | 4 functions | 4 (linear chain) | Deep interprocedural |
| F012 | `callgraph-deep-8` | 8 functions | 8 | Very deep call chain |

### F.4 Loop nest scaling (F013–F016)

| ID | Name | Nesting depth | Trip counts | Purpose |
|----|------|---------------|-------------|---------|
| F013 | `nest-2-deep` | 2 | 64 × 64 | Baseline nested |
| F014 | `nest-3-deep` | 3 | 32 × 32 × 32 | Cost model trip product |
| F015 | `nest-4-deep` | 4 | 16 × 16 × 16 × 16 | Deep nest analysis |
| F016 | `nest-mixed-affine-scf` | 2 | 64 × dynamic | Affine outer, SCF inner |

### F.5 Pathological cases (F017–F020)

| ID | Name | Description | Purpose |
|----|------|-------------|---------|
| F017 | `diamond-dag-wide` | Binary DAG with depth 8 (255 nodes) | SSA fanout explosion |
| F018 | `self-referential-view` | Deeply chained view-of-view-of-view (8 hops) | View resolution cost |
| F019 | `many-consumers` | 1 store, 64 loads in separate blocks | Consumer iteration stress |
| F020 | `interleaved-allocs` | 32 allocs with interleaved store/load patterns | Provenance map stress |

---

## G. Cross-Feature Interaction (G001–G020)

Programs that test interactions between DataRecomputation, MemoryFission, the cost
model, and buffer elimination working in concert.

### G.1 DataRecomputation + MemoryFission (G001–G006)

| ID | Name | Description | Expected |
|----|------|-------------|----------|
| G001 | `dr-fission-agree` | DR says recompute, fission would also recompute | Both agree: recompute |
| G002 | `dr-fission-disagree` | DR says recompute (small buffer), fission says keep (many consumers) | Depends on pass order |
| G003 | `dr-then-fission` | DR eliminates some loads; fission sees fewer consumers | Pass ordering matters |
| G004 | `fission-then-dr` | Fission materializes to buffer; DR could eliminate that buffer | Reverse ordering test |
| G005 | `dr-fission-nested` | DR handles inner loop, fission handles outer | Orthogonal optimizations |
| G006 | `dr-fission-shared-chain` | Same chain targeted by both passes | Priority/conflict |

### G.2 Cost model ambiguity zones (G007–G012)

Programs where the cost model decision is close to break-even and small parameter
changes flip the outcome.

| ID | Name | Description | Flip trigger |
|----|------|-------------|--------------|
| G007 | `ambig-l1-boundary` | Buffer 31KB, ALU=4, 1 consumer | ±1KB flips L1/L2 boundary |
| G008 | `ambig-consumer-flip` | 2 consumers at break-even | Adding 3rd consumer flips |
| G009 | `ambig-alu-tie` | ALU cost = load latency exactly | Any perturbation flips |
| G010 | `ambig-footprint-evict` | Intervening footprint ≈ L1 size | ±4KB changes eviction estimate |
| G011 | `ambig-stride-cost` | Stride puts leaf cost at partial-remat threshold | ±1 stride flips |
| G012 | `ambig-json-override` | Default says recompute, JSON cost model says keep | Tests override priority |

### G.3 Buffer elimination + strategies (G013–G016)

| ID | Name | Description | Expected |
|----|------|-------------|----------|
| G013 | `elim-after-remat` | DR replaces 3/4 loads; buffer elim evaluates remaining 1 | Partial→full elimination |
| G014 | `elim-blocked-by-multi` | 3 SINGLE + 1 MULTI load; DR replaces 3, MULTI blocks elim | Cannot fully eliminate |
| G015 | `elim-drives-override` | Per-load cost says keep, but whole-buffer rollup says eliminate | `dr-buffer-elim-drives-strategies` flag |
| G016 | `elim-partial-remat-leaf` | Buffer elim feasible only if partial remat handles leaf loads | Strategy dependency |

### G.4 Full pipeline compositions (G017–G020)

Programs that exercise the complete `raise-malloc → data-recomputation → memory-fission`
pipeline as used in SPEC/ONNX compilation.

| ID | Name | Description | Purpose |
|----|------|-------------|---------|
| G017 | `pipeline-matmul-like` | 3 nested loops: init buffer, compute, reduce | Realistic pattern |
| G018 | `pipeline-stencil-like` | 1D stencil: neighbors loaded, result stored | Spatial reuse pattern |
| G019 | `pipeline-conv-like` | 2D convolution kernel with weight buffer | ONNX-like pattern |
| G020 | `pipeline-reduce-like` | Reduction over large input into small output | Accumulator pattern |

---

## Scaled Variant Naming

Programs with scaling variants use suffixes:

- Buffer size: `-4kb`, `-32kb`, `-256kb`, `-64mb`
- Trip count: `-trip8`, `-trip64`, `-trip1024`
- Chain depth: `-chain2`, `-chain8`, `-chain64`
- Consumer count: `-cons1`, `-cons4`, `-cons8`
- Call depth: `-depth1`, `-depth2`, `-depth3`
- Stride: `-stride1`, `-stride4`, `-stride16`
- Nesting: `-nest2`, `-nest3`, `-nest4`

Example: `bench/C/C002-sweep-l1-boundary-30kb.mlir`, `...-32kb.mlir`, `...-34kb.mlir`

The base ID count (200) includes all scaled variants. A program with 3 scale
points counts as 3 toward the 200.

---

## RUN Line Templates

Every program uses one of these `// RUN:` patterns:

### Classification test (diagnostic)
```mlir
// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics
```

### Recomputation test (transform + FileCheck)
```mlir
// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute})' | FileCheck %s
```

### Cost model test (transform gated by cost)
```mlir
// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-l1-size=32768 dr-l2-size=262144})' | FileCheck %s
```

### Partial remat test
```mlir
// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-partial-remat dr-partial-max-leaves=4})' | FileCheck %s
```

### Footprint analysis test
```mlir
// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-footprint-analysis})' | FileCheck %s
```

### Buffer elimination test
```mlir
// RUN: dr-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-buffer-elim})' -verify-diagnostics
```

### Memory fission test
```mlir
// RUN: dr-opt %s --pass-pipeline='builtin.module(memory-fission{l1-size=32768 l2-size=262144})' | FileCheck %s
```

### Full pipeline test
```mlir
// RUN: dr-opt %s --pass-pipeline='builtin.module(raise-malloc-to-memref,data-recomputation{dr-recompute dr-cost-model},memory-fission)' | FileCheck %s
```

---

## Implementation Order

### Phase 1: Scaffolding (1 session)
1. Create directory structure: `bench/{A,B,C,D,E,F,G}/`
2. Create `bench/lit.cfg.py` extending existing lit config
3. Write a generator script `bench/gen.py` for parameterized variants

### Phase 2: Classification & Strategy (2 sessions)
4. Implement A001–A030 (classification stress)
5. Implement B001–B040 (strategy selection)

### Phase 3: Cost Model & Fission (2 sessions)
6. Implement C001–C040 (cost model boundaries)
7. Implement D001–D030 (memory fission)

### Phase 4: Elimination & Scaling (1 session)
8. Implement E001–E020 (buffer elimination)
9. Implement F001–F020 (scaling stress)

### Phase 5: Interaction & Validation (1 session)
10. Implement G001–G020 (cross-feature interaction)
11. Run full suite, fix broken tests, verify coverage

### Phase 6: Analysis (1 session)
12. Run suite with `dr-summary` to collect decision statistics
13. Identify programs where cost model gives wrong/surprising answer
14. Document findings: strengths, weaknesses, ambiguity zones

---

## Success Criteria

- All 200 programs pass `llvm-lit` (correct classification/transform/skip)
- Every pass option exercised by at least 3 programs
- Every strategy exercised by at least 5 programs
- Cost model boundary programs produce different decisions when cache params change
- Memory fission programs verify both "fission" and "no fission" outcomes
- Buffer elimination programs cover all `EscapeKind` variants
- Scaling programs complete in < 10s each (compile-time regression gate)
- At least 10 programs identified where cost model gives ambiguous/wrong answers (documented)

---

## Measurement & Reporting

After implementation, run:

```bash
# Full suite correctness
llvm-lit bench/ -v --timeout 30

# Cost model decision dump
for f in bench/C/*.mlir bench/G/*.mlir; do
  dr-opt "$f" --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model dr-summary})' 2>&1 | grep DRSUM
done > bench/cost-model-decisions.log

# Compile-time profiling
for f in bench/F/*.mlir; do
  time dr-opt "$f" --pass-pipeline='builtin.module(data-recomputation{dr-recompute dr-cost-model})' > /dev/null
done 2> bench/compile-times.log
```
