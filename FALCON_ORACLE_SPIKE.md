# Falcon-as-oracle spike

Question: is drcompiler's cache cost model wrong in ways an exact analytical
cache model would fix, and do those errors change transformation decisions?

Oracle: **Falcon** (Pitchanathan, Grover, Grosser, PLDI 2024), artifact
Zenodo 10972076.  Its tool is called `lazystack` and takes **affine MLIR**
directly.

## 0. Integration feasibility — the actual go/no-go

The `lazystack` binary (42 MB, in `cmake-build-release/bin/`) **runs natively
on this machine with no missing shared libraries**.  The 6.2 GB Docker image
in the artifact is not needed.

Its input corpus (`benchmark/polybench/mlir/{S,M,L,XL}/*.mlir`) is
Polygeist/cgeist affine MLIR, byte-for-byte the same shape as
`bench/polybench-mlir/` — same `func.func @kernel_gemm`, same
`memref<?x1100xf64>`, same `affine.for` — except the PolyBench dataset sizes
are constant-folded into the loop bounds.  The only edit needed in either
direction is the DLTI `vector<Nxi32>` <-> `vector<Nxi64>` fixup that `drcc`
already applies (LLVM 18 vs 22 serialization).

Invocation:

```
lazystack --cs=512 -a 512 --cs=8192 -a 8192 --line-size=64 -n 8 kernel.mlir
```

= two fully-associative levels, 64 B lines, L1 32 KiB / L2 512 KiB, matching
drcompiler's `MachineModel` defaults.  Output is JSON with `accesses`,
`misses`, `misses_L1`, `misses_L2`.  Runtime 25 s to a few minutes per
PolyBench-L kernel.

**Verdict: integration is cheap.** A binary that accepts our IR unmodified,
on our benchmark corpus, at seconds-to-minutes per query.

## 1. The experiment

`bench/falcon-spike/oracle-vs-model.sh`, over all 30 PolyBench-L kernels:

1. `lazystack` on the untiled nest -> baseline miss count
2. `dr-affine-loop-distribute,dr-affine-loop-tile{emit-rationale}` -> verdict
   (the distributor is required ahead of the tiler: Polygeist emits imperfect
   nests, and the tiler only sees top-level perfect bands)
3. `lazystack` on the transformed nest -> post-transform miss count

A model doing its job TILEs exactly the kernels whose miss count improves.

Raw data: `bench/falcon-spike/results.csv`.

## 2. Where the model is right

| kernel | verdict | misses before | misses after | change |
|---|---|---:|---:|---:|
| seidel-2d | TILE | 999,250,000 | 1,110,500 | **900x fewer** |
| 2mm | TILE | 2,071,901,600 | 221,704,000 | **9.3x fewer** |
| 3mm | TILE | 3,377,715,400 | 371,633,200 | **9.1x fewer** |
| floyd-warshall | TILE | 2,743,983,968 | 1,595,377,121 | 1.72x fewer |
| gemm | TILE | 331,776,000 | 193,680,800 | 1.71x fewer |

Five of eight TILE decisions are real, and seidel-2d is spectacular.  This is
worth stating plainly: **the cost model's positive decisions are mostly
sound**, and an oracle would not have found more on these.

## 3. Where the model is wrong

### False positives — TILE with no benefit

| kernel | verdict | misses before | misses after | change |
|---|---|---:|---:|---:|
| covariance | TILE | 2,139,306,482 | 2,138,948,706 | 1.0002x (nothing) |
| correlation | TILE | 2,139,517,386 | 2,139,161,267 | 1.0002x (nothing) |
| trmm | TILE | 1,200,434,681 | 1,200,734,681 | 0.9998x (slightly worse) |

Three of eight TILE decisions buy nothing.  Note covariance and trmm are two
of the campaign's headline speedups (11.55x, 16.1x in
`claude-docs/COSTMODEL_V3_FINDINGS.md`) — but those came from other
mechanisms ("A2 k-chunked peeled HEAD", "A3 in-place triangular peel"), not
from this tiling.  The tiling itself is a no-op for cache misses.

### The dominant failure is coverage, not precision

Of 22 REJECTs, **17 are `reason=out-of-model`** — "non-constant trips,
imperfect below band, or non-affine references".  Only 5 are an actual
judgement (`no-evicted-reuse`: atax, bicg, gemver, mvt, and one band each in
a couple of others).

The model does not mostly make *wrong* decisions.  It mostly **declines to
decide**, because `analyzeBandReuse` bails.  The single largest cause is nest
imperfection: Polygeist emits an init loop beside the compute loop, and the
ping-pong stencils (jacobi-2d, heat-3d, fdtd-2d, adi) put two sibling space
nests under the time loop, so the top-level band is depth-1 with everything
interesting below it.

This reframes the Presburger question from the earlier analysis.  Exact set
arithmetic addresses trip counts and reference precision — but those are not
what is binding.  What is binding is that 17 of 30 kernels never reach the
cost model at all.

### Were the REJECTs right? Force-tiling says: almost all, yes

`bench/falcon-spike/forced-tile.sh` re-runs each big REJECT with the cost
model bypassed (`-tile-size=32`) and measures the miss count again.  If a
rejection were a false negative, forced tiling would improve on it.

Of 14 rejected kernels force-tiled, **13 show no change at all** (adi,
cholesky, fdtd-2d, gramschmidt, heat-3d, jacobi-2d, lu, ludcmp, nussinov,
syr2k, syrk, doitgen, deriche — identical or within a few dozen misses).  The
rejections are harmless: the tiler cannot act on those nests regardless of
what the cost model says, because the band it can see is a depth-1 outer loop.

The exception is **symm**, and it is a genuine false negative:

| symm | total misses | L1 | L2 |
|---|---:|---:|---:|
| baseline (REJECT, untouched) | 1,360,626,125 | 1,210,520,699 | 150,105,426 |
| forced `tile-size=32` | 1,218,387,669 | 1,210,082,021 | **8,305,648** |
| | 1.12x fewer | — | **18.1x fewer** |

The model rejected symm with `reason=out-of-model`, and tiling it would have
cut L2 misses by 18x.  One real miss out of 30 kernels — and again the cause
is coverage (the band was never analyzable), not a mis-costed decision.

### Splitting the out-of-model rejections by actual cause

`bench/falcon-spike/classify-rejects.sh` runs the distributor, then reports per
kernel how many bands `analyzeBandReuse` refuses, how many loops carry a
symbolic bound (`to #map(%iv)`), and the depth of the band the tiler actually
sees.  Of the 18 kernels checked:

| cause | count | kernels |
|---|---:|---|
| symbolic / triangular bound present | 10 | cholesky, durbin, gramschmidt, lu, ludcmp, nussinov, symm, syr2k, syrk, trisolv |
| ping-pong stencil, top band depth 1, no symbolic bound | 5 | adi, fdtd-2d, heat-3d, jacobi-1d, jacobi-2d |
| other (non-affine refs / refs outside band) | 3 | deriche, doitgen, gesummv |

symm, syr2k, syrk and nussinov reach depth 3 — deep perfect-looking bands
refused on the bound, not on nest shape.  symm is additionally imperfect: an
`affine.store` to a scalar `alloca` accumulator sits between the `j` and `k`
loops, and there is a tail after the `k` loop.  Running
`dr-scalar-reduction-demote` ahead of distribute+tile does **not** rescue it —
still `out-of-model`.

This corrects §6's first framing.  The coverage gap is not purely structural:
its largest slice (10 of 18) is symbolic trip counts, which *is* the
Presburger-adjacent part.  Note though that the *analyzability gate* probably
does not need Presburger — for `ub = #map(%i)` with `%i` ranging over a
constant band loop, a max-trip substitution (`i + 1 <= 1200`) is enough to
admit the band.  Exact counting would additionally remove the 2x footprint
over-count on triangular domains (§5), which is a separate, smaller benefit.

### Distribution can make things worse unnoticed

gramschmidt: REJECT (no band tiled), yet misses rose 2,971,913,015 ->
3,062,489,952, with **L2 misses doubling** (90,622,415 -> 181,019,659).  That
is `dr-affine-loop-distribute` acting alone.  Nothing in the pipeline models
the cost of the distribution itself.

## 4. A methodological correction

An earlier iteration of this spike used a hand-written corpus
(`bench/falcon-spike/kernels.mlir`) in which the stencil kernels were reduced
to a single space step, with the time loop dropped.  On that corpus all three
stencils were rejected with `no-evicted-reuse`, and I concluded the model was
categorically blind to stencil reuse because `loopCarriesEvictedReuse` only
considers references that are `invariantIn(loopIdx)` and a stencil neighbour
`A[i][j-1]` is invariant in no space loop.

**The real corpus refutes that conclusion.**  With the time loop present the
stencil references *are* invariant in `t`, the model sees the reuse, and
seidel-2d tiles to a 900x miss reduction — the single best decision it makes
anywhere.  The failure on jacobi-2d / heat-3d / fdtd-2d / adi is nest
imperfection (ping-pong buffers), not the group blind spot.

The `RefGroup` blind spot is real as a code fact — `info.groups` has exactly
one consumer in the tree, the test printer, and `ReuseAnalysis.h` says no
verdict consults it — but it does not bind in practice, because the time loop
supplies the invariance instead.  Dropping the time loop tested a shape that
does not occur in the benchmark corpus.

## 5. Footprint accuracy (still true, now in context)

Measured on the hand-written corpus against enumerated distinct cache lines
(`bench/falcon-spike/ground_truth.py`):

| kernel | model footprint | true | ratio |
|---|---:|---:|---:|
| gemm | 28.96 MB | 28.96 MB | 1.00x |
| trmm | 27.20 MB | 13.62 MB | 2.00x |
| covariance | 38.40 MB | 19.24 MB | 2.00x |
| jacobi-2d | 80.87 MB | 27.02 MB | 2.99x |
| heat-3d | 105.15 MB | 27.19 MB | 3.87x |
| seidel-2d | 103.33 MB | 11.52 MB | 8.97x |

`footprintBytes` sums a per-reference bounding box, so references to the same
array are counted repeatedly; the control (distinct arrays) is exact.

Separately, `refFootprint` is `elemBytes * product-of-extents` with no
rounding to cache lines, so a column-strided f64 reference is under-counted
8x.  That breaks the safety argument written at
`lib/Transforms/AffineLoopTilingCostModel/LoopTiling.cpp:229` ("over-estimated
footprints only err toward tiling") — the line error errs the other way.

These are real defects.  I guessed they were why covariance/correlation/trmm
tile for nothing.  **That guess was wrong — see §7.**

## 6. What this means for the campaign

**Consuming Falcon is cheap and works.**  §0 settles the integration risk that
the earlier analysis flagged as the main unknown: no Docker, no C extraction,
no isl plumbing, our IR accepted as-is.

**But the headline result is not the one expected.**  The hypothesis was that
an exact oracle would flip decisions the approximate model gets wrong.  What
the data shows is:

- the model's *positive* decisions are mostly right (5/8), and one of them is
  worth 900x;
- its *wrong* positives (3/8) are a real but bounded problem;
- its rejections are almost all harmless (13/14 force-tile to no change);
  exactly one, symm, is a real lost 18x on L2 misses;
- its dominant behaviour is refusing to decide at all (17/30 out-of-model),
  which an oracle does not fix, because the oracle is queried per-band and the
  bands never form.

So the ordering is: **fix coverage first, then consider the oracle.**  Per the
cause breakdown above, that means two independent pieces of work — admit
symbolic trip counts (10 kernels, contains the only confirmed lost win) and
handle sibling nests under a shared time loop (5 stencils) — before decision
*quality*, which is what an oracle improves, becomes the binding constraint.

The oracle's clearest immediate use is not as a gate but as a **regression
check**: it catches gramschmidt's silent 3% / 2x-L2 regression, and it catches
covariance/correlation/trmm tiling for nothing, neither of which any existing
test would notice.

## 7. Follow-up: the footprint defects were fixed, and it changed nothing

Four fixes landed after the spike, each validated against the enumerated
ground truth:

1. **Symbolic (triangular) bound admission** — `analyzeBandReuse` resolves
   `0 to #map(%i)` by substituting the enclosing band loop's range
   (`constantIvUpperBound`).  Also fixed a real bug found on the way: the
   upper-bound path never divided by the loop step, so a stepped triangular
   loop reported 4x its trips.
2. **Union same-array references** — `footprintBytes` is a *capacity* figure,
   so references to one memref are now hulled per dim instead of summed.
   `refFootprintBytes` still sums, because traffic counts streams.
3. **Cache-line rounding** — the fastest-varying subscript's byte run rounds
   up to whole lines (`BandReuseInfo::cacheLineBytes`, set from the machine
   model), so a column-strided f64 costs a line per row, not an element.
4. **Divisor-snap floor** (`LoopTiling.cpp`) — snapping a tile size down to a
   trip-count divisor used an unbounded decrement, so trip 398 with a chosen
   tile of 16 collapsed to **2**, a quarter of a cache line.  Now floored at
   half the chosen size.

Footprints afterwards, against enumeration: heat-3d and seidel-2d **exact**,
jacobi-2d within 0.3%, gemm within 0.2%, trmm and covariance improved from
2.00x over to 1.29x / 1.30x (residual is the hull being loose across
*differing* coefficient matrices).  The 8x column under-count is gone:
covariance's `data[k][i]` reuse distance at loop 1 is now 89,600 B against an
enumerated 89,600 B.

**Measured effect on the compiler: none.**  Full PolyBench-L sweep, all 30
kernels, before vs after: 0 worse, 0 better, 30 unchanged.  The only verdict
changes are syrk and syr2k moving REJECT -> TILE with byte-identical miss
counts, i.e. two more inert tilings.

So the §5 hypothesis is **falsified**: covariance, correlation and trmm still
tile for nothing with footprints that are now near-exact.  Footprint error was
not the cause.  Two candidates remain untested — the traffic objective picks a
bad tile *shape*, or these kernels genuinely cannot benefit from tiling at a
256 KiB target and the correct verdict is REJECT.

One open regression, at M size only: seidel-2d 40,000 -> 2,980,000 misses.  At
L it is untouched at the full 900x.  Cause: nothing penalises the reuse
distance across an untiled loop whose subscript coefficient is zero (the time
loop), so when the capacity constraint is slack the search leaves it untiled
(`sizes=[100, 398, 16]`).  At L the 32 MB working set forces `[500, 9, 9]` and
the right answer falls out anyway.  The previously-reported claim that
seidel-2d's 900x was an artifact of the double-counting bug was wrong — it was
measured at M only and does not hold at L.

Net: four genuine correctness fixes, near-exact footprints, no performance
change at evaluation scale.  The model was wrong in ways that did not affect
its decisions — which is itself the useful result, and the opposite of what
this spike set out to show.

## 8. The inert tilings are correct REJECTs, and the model structurally cannot know

`bench/falcon-spike/tile-shape-sweep.sh` force-tiles each inert kernel at a
spread of uniform sizes, bypassing the cost model, and measures misses at L:

| kernel | untiled | ts=16 | ts=32 | ts=64 | ts=128 | best |
|---|---:|---:|---:|---:|---:|---:|
| correlation | 2,139,517,386 | 2,139,187,416 | 2,139,174,234 | 2,139,174,004 | 2,139,167,404 | 1.000x |
| covariance | 2,139,306,482 | 2,138,961,790 | 2,138,955,224 | 2,138,955,133 | 2,138,951,841 | 1.000x |
| trmm | 1,200,434,681 | 1,200,734,681 | 1,200,734,681 | 1,200,734,681 | 1,200,734,681 | 1.000x |
| syrk | 796,228,019 | 796,370,027 | 796,370,027 | 796,370,027 | 796,370,027 | 1.000x |
| syr2k | 1,655,759,480 | 1,655,879,352 | 1,655,879,352 | 1,655,879,352 | 1,655,879,352 | 1.000x |

**No size helps any of them.** Every point is within 0.02% of untiled and
several are marginally worse.  So the tiling is not badly shaped — tiling
cannot help these kernels at this cache configuration, the correct verdict is
REJECT, and the gate is too permissive.  Five of eight TILE decisions are
wrong, and they cost loop-restructuring overhead for nothing.

Caveat: this tests uniform sizes plus the model's own (non-uniform) choice, not
the full shape space.  Strong evidence, not proof.

### Why the model cannot tell

The tile search minimises `traffic = Σ_refs weight × tiles × refFootprint(tile)`
(`LoopTiling.cpp`).  `tiles` is the product of `ceil(trip / tileSize)`, so the
objective is monotonically increasing as tiles get smaller: **untiled is always
its minimum.**  The metric can therefore only rank candidate tilings against
each other — it can never say "none of these beats not tiling".

That job falls entirely to the separate `loopCarriesEvictedReuse` gate, which
asks *"is there temporal reuse whose distance exceeds the cache?"* — not
*"would tiling actually fix it?"*.  For these five kernels the answer to the
first is yes and to the second is no, and nothing in the model closes the gap.

Fixing this needs a profitability metric comparable across tile counts (a
modelled miss count, not a traffic sum), which is a design change rather than a
tuning fix — and, notably, exactly what an oracle provides directly.  This is
the strongest argument in this document for the oracle-as-gate idea, arrived at
only after the cheaper explanations were measured and eliminated.

## 9. Harness: per-kernel dataset sizes

One global dataset size produced a verdict that did not survive (§7's
retraction).  `bench/falcon-spike/pick-sizes.sh` now chooses a size per kernel,
requiring **both**:

1. L2 capacity misses exist in the untiled kernel (oracle-measured) — else the
   working set fits and tiling cannot help, so the size proves nothing;
2. the band's working set exceeds the tiler's target by >= 20x — the condition
   that actually forces the tile search.  Condition 1 alone still picked M for
   seidel-2d, where the set is only 4.9x over target.

Result: 8 kernels at S, 9 at M, 13 at L; seidel-2d correctly at L.  atax and
bicg satisfy neither at any size (working set 0.1x target) and are flagged
`slack-capacity` — tiling can never help them, so their rows only guard against
regressions.

`cache-regress.sh` reads the manifest and runs kernels concurrently: the full
30-kernel sweep takes **39 s** despite including 13 L-size kernels.  Verified
both ways — clean on the real pipeline (exit 0), and exit 1 with
floyd-warshall +72%, gemm +71%, seidel-2d +89882% when pointed at a
deliberately worse pipeline.

## 10. Tier 0: the coverage gap was band ENUMERATION, and closing it found symm

§6 attributed 17 of 22 REJECTs to being out-of-model and split the causes by
reading the IR: 10 symbolic/triangular bounds, 5 ping-pong stencils, 3 other.
That split was wrong.  Attributing every `failure()` return in
`analyzeBandReuse` to a named `ReuseReject` and running it over PolyBench-L
gives the real histogram, and it has exactly one entry:

| cause | bands |
|---|---:|
| `non-band-operand` | 26 |
| everything else | 0 |

Not one trip-count rejection, not one symbolic subscript, not one semi-affine
subscript.  All 26 were the same defect wearing a misleading name.

**What `non-band-operand` really was.**  Band enumeration used
`getPerfectlyNestedLoops` from a top-level `affine.for`.  On the ping-pong
shape `for %t { for %i { for %j {A} }  for %i { for %j {B} } }` that truncates
the band to `[%t]`, because `%t`'s body holds two loops.  The analysis then
walked `%t`'s whole subtree and rejected the first subscript it saw — every one
of them names the `%i`/`%j` that truncation had just excluded.  The band it
complained about was one it built itself.  The two `[%i, %j]` bands were always
inside the constant-coefficient model; nothing ever looked at them.

**Three restrictions, all the same shape.**  A band's context was invisible to
it in both directions:

1. *Enumeration* — `collectMaximalPerfectBands` reports every maximal perfect
   band, including bands under an imperfect enclosing loop.
2. *Enclosing IVs* — fixed for the band's whole execution, so they shift a
   base address and widen no extent (`RefInfo::outerCoeff`).  Seeding the bound
   map with them also resolves inner triangular bounds like `for %j = %i+1 to N`.
3. *Inner IVs* — a loop below the band sweeps its full range every band
   iteration, so it widens extents by a tile-shape-independent constant
   (`RefInfo::innerExtent`).

Cumulative effect on rejected bands, PolyBench-L:

| | rejected bands | kernels with NO analyzable band |
|---|---:|---:|
| before | 26 | 15 of 30 |
| + enumeration (1) | 70 | 10 of 30 |
| + enclosing IVs (2) | 34 | 0 of 30 |
| + inner IVs (3) | **5** | **0 of 30** |

The band count rises at step 1 because far more bands are now enumerated; the
kernel column is the one to read.  The 5 survivors are honest: 2 are nussinov's
genuinely non-affine accesses (`opaque-op`), 3 are `non-band-operand` on
subscripts naming a value that is neither an enclosing nor an inner IV.

**And it changed almost nothing — with one exception that matters.**  Tile
verdicts over the corpus stayed at 13 TILEs across the same 10 kernels, and the
oracle reports 0 regressions and the same 5 pre-existing inert tilings.  This is
the third independent confirmation of §7/§8: precision was harmless, tile shape
was harmless, and now coverage is *almost* harmless too.

The exception is **symm**, the one kernel §6 flagged as a real false negative:

```
symm @ M   REJECT -> TILE   L2 misses 657,592 -> 27,200   (24.2x fewer)
```

That is a genuine win the model could not previously reach, and it came from
(3) — symm's `[%i, %j]` band was already top-level and perfect; it was rejected
only for naming the `%k` of its own reduction loop.

**symm at L is still wrong, and now we know exactly why.**  At L the analysis
handles the band correctly (`evictedReuse=[1,0]`, so it does see the reuse),
and the *tile search* rejects with `no-feasible-tile`.  Forcing the tile proves
the search is wrong to:

| symm @ L | L2 misses | vs untiled |
|---|---:|---:|
| untiled | 150,105,426 | — |
| forced tile 16 | 11,083,018 | 13.5x fewer |
| **forced tile 32** | **8,305,648** | **18.1x fewer** |
| forced tile 64 | 112,592,546 | 1.3x fewer |

The fit test requires the whole tile footprint to fit the target, and symm's
tile is dominated by `B[%k][%j]`, which is *streamed*, not reused — it never
needs to be resident.  Requiring residency for streamed references is what
makes an 18.1x tiling look infeasible.  This is the same structural defect §8
named from the other side: the gate reasons about a footprint sum rather than a
modelled miss count.  It is now sharply posed — *only references carrying the
reuse need to fit* — and it is the first concrete lead on that defect that
comes with a measured payoff attached.

**Scope note.**  Enumeration (1) is opt-in on both passes
(`dr-affine-loop-tile{all-bands=true}`, `dr-test-reuse-analysis{all-bands=true}`)
because it is measurably decision-neutral today; (2) and (3) are unconditional,
which is where symm's win comes from.  Turning (1) on becomes worthwhile the
moment the fit test above is fixed, since that is what makes stencil inner
bands actionable.

```
bench/falcon-spike/reject-histogram.sh      # why analyzeBandReuse bails, per band
bench/falcon-spike/rationale-histogram.sh   # what the tile gate then decides
```

## References

- Pitchanathan, Grover, Grosser. *Falcon: A Scalable Analytical Cache Model.*
  PLDI 2024. <https://dl.acm.org/doi/10.1145/3656452>
- Artifact: <https://zenodo.org/records/10972076>
- Gysi, Grosser, Brandner, Hoefler. *A Fast Analytical Model of Fully
  Associative Caches* (HayStack). PLDI 2019.

## Repro

```
bench/falcon-spike/cache-regress.sh                      # regression gate (39 s)
bench/falcon-spike/cache-regress.sh --update             # regenerate baseline.csv
bench/falcon-spike/pick-sizes.sh sizes.csv               # per-kernel size manifest
bench/falcon-spike/tile-shape-sweep.sh <outdir> <k>...   # does ANY tile size help?
bench/falcon-spike/oracle-vs-model.sh <outdir>           # full L sweep -> results.csv
bench/falcon-spike/forced-tile.sh <outdir> <kernel>...   # score a REJECT
bench/falcon-spike/classify-rejects.sh <outdir> [k...]   # why did a band bail?
bench/falcon-spike/ground_truth.py                      # footprint ground truth
bench/falcon-spike/reuse_distance.py                    # reuse-distance ground truth
bench/falcon-spike/reject-histogram.sh                   # §10 analysis coverage
bench/falcon-spike/rationale-histogram.sh                # §10 tile-gate verdicts
```
