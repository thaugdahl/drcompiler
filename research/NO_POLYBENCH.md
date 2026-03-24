# Why PolyBench Is Not a Good Target for drcompiler

Analysis of whether extending DataRecomputation or MemoryFission to handle
PolyBench/C 4.2.1 patterns is worth the effort. Conclusion: it is not.

## The Fundamental Mismatch

PolyBench kernels are dominated by **reductions and accumulations** where the
stored intermediate value is the result of an O(N) inner loop. DataRecomputation
replaces loads with recomputation of the stored value — but recomputing a
reduction is always more expensive than a cache miss.

| Kernel      | Intermediate   | How It's Computed                          | Recompute Cost     |
|-------------|----------------|--------------------------------------------|--------------------|
| GEMM        | `C[i][j]`      | `sum_k(A[i][k]*B[k][j])`                  | O(NK) per element  |
| 2MM         | `tmp[i][j]`    | `sum_k(A[i][k]*B[k][j])`                  | O(NK) per element  |
| Correlation | `mean[j]`      | `sum_i(data[i][j]) / N`                    | O(N) per element   |
| Correlation | `stddev[j]`    | `sqrt(sum_i((data[i][j]-mean[j])^2) / N)` | O(N) per element   |
| Deriche     | `y1[i][j]`     | recurrence on `y1[i][j-1], y1[i][j-2]`    | O(j) per element   |
| Jacobi-2D   | `B[i][j]`      | 5-point stencil (cheap, but read 5x each)  | 10 ops × 5 reads   |

Even with perfect analysis, the cost model would reject every candidate.

## DataRecomputation: Three Layers of Failure

### Layer 1: Most Hot Loads Are Read-Only Inputs

In GEMM, the bandwidth-critical loads are `A[i][k]` and `B[k][j]` — input
arrays with no stores in the kernel. DataRecomputation has nothing to forward
because there is no store-to-load chain. This applies to most PolyBench kernels.

### Layer 2: Accumulations Break Provenance

The stores that do exist (`C[i][j] += ...`) create loop-carried dependences.
Tracing the analysis for GEMM's k-loop:

1. The i/j loops have symbolic bounds (PolyBench macros) so `joinStoreMaps`
   merges pre-loop and body state at each loop exit.
2. The k-loop body's `affine.load %C[%i,%j]` sees one store op (from the
   pre-k-loop `C *= beta`) and gets classified **SINGLE**.
3. But `isRematerializable` fails: the stored value depends on
   `affine.load %C[%i,%j]` from before the beta-scaling, which has **MULTI**
   provenance from the outer i/j loop joins.

The chain breaks at the outer loop boundary regardless of index precision.

### Layer 3: The Index Tracking Gap Does Not Matter Here

Currently `affine.store`/`affine.load` use `std::nullopt` coverage (conservative).
Adding `mlir::affine::checkMemrefAccessDependence` for precise per-element
coverage would be ~2-3 weeks of engineering. But on PolyBench it would prove
things that don't lead to profitable transformations:

- Read-only arrays have no stores (nothing to prove).
- Accumulated arrays get MULTI provenance from outer loop joins, not from index
  aliasing.
- The one thing it would enable — proving `store [%i]` and `load [%j]` don't
  alias when `%i != %j` — only matters when there are multiple store ops to the
  same array at different indices within a single loop body, which PolyBench
  kernels rarely have.

## MemoryFission: Wrong Abstraction

The pass finds sibling simple affine loops with **structurally identical
expensive computations** from the same source memref. PolyBench has none of
this:

- No duplicate computations across sibling loops.
- All interesting loops are multiply-nested (inner `affine.for` triggers bail-out).
- 2MM's two matmuls read different source arrays (`A*B` vs `tmp*C`), so they
  don't share a fingerprint.

The optimization that would help PolyBench is **classical loop distribution** —
splitting a loop body touching multiple arrays into separate loops with smaller
working sets. That is a fundamentally different transformation requiring
statement-level dependence analysis and working-set partitioning. Implementing
it would be ~4-8 weeks and would replicate existing polyhedral transforms
(PLUTO, ISL-based distribution).

## What Actually Speeds Up PolyBench

The known optimizations for PolyBench are standard polyhedral transforms, most
of which are available as built-in MLIR passes:

| Transform          | MLIR Pass                  | Impact                   |
|--------------------|----------------------------|--------------------------|
| Loop tiling        | `affine-loop-tile`         | High (cache blocking)    |
| Loop interchange   | `affine-loop-interchange`  | High (stride optimization) |
| Scalar replacement | `affine-scalrep`           | Medium (register promotion) |
| Vectorization      | `affine-super-vectorize`   | High (SIMD)              |

These are what Polly, PLUTO, and other polyhedral compilers use on PolyBench.

## Fleetbench Is Also Not Viable

Google Fleetbench was considered as an alternative. It fails on a more basic
level: **pipeline incompatibility**.

- C++ with heavy template/STL usage — cgeist cannot lower protobuf, abseil, gRPC.
- Opaque library calls — hot code (Snappy, CRC32, absl::Hash) is pre-compiled;
  dr-opt never sees the inner loops.
- Bazel build system — drcc is a drop-in `cc` for Make/CMake builds.
- Microbenchmarks — each benchmark invokes a single hot function in a measurement
  loop with no interprocedural data flow to analyze.

## Where drcompiler's Passes Do Apply

DataRecomputation and MemoryFission target patterns found in **irregular,
interprocedural C code** — not regular dense linear algebra:

- **DataRecomputation**: helper functions that compute a value and write it to an
  output parameter, where the caller loads it later. Scalar/small-buffer
  store-then-load chains where the expression is cheap relative to the cache
  miss. These patterns appear in SPEC CPU 2017 C benchmarks (`lbm`, `mcf`,
  `nab`, `imagick`, `x264`).
- **MemoryFission**: duplicate expensive pointwise computations (sqrt, div,
  transcendentals) shared across multiple consumer loops. More likely in
  scientific simulation codes with repeated physics kernels.

SPEC CPU 2017 remains the right evaluation target.
