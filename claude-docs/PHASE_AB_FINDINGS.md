# Generalized Detector (a) + Cache Tiling (b) — `affine-register-block`

**Date:** 2026-06-04
**Author:** Claude (Opus 4.8)
**Status:** Both landed. Pass now handles real PolyBench-shaped GEMM and recovers
large-N performance via cache blocking.

Builds on `PHASE1_FINDINGS.md` (the perfect-nest clean-gemm gate) and
`COSTMODEL_SPIKE_FINDINGS.md` (fixed register tile).

---

## (a) Generalized detector — imperfect nests + i-k-j reduction order

**Problem.** The Phase-1 detector required a perfect 3-nest with the reduction
loop innermost. The *real* PolyBench gemm is neither:
```
for i:                       # i imperfectly nests two children:
  for j: C[i,j] *= beta      #   (1) a beta-scaling sibling loop
  for k:                     #   (2) the matmul, in i-k-j order:
    for j: C[i,j] += alpha*A[i,k]*B[k,j]   # reduction k is the MIDDLE loop,
                                           # accumulator varies in innermost j
```

**Fix (two new stages in the pass):**
1. **Canonicalization** (`canonicalizeOnce`): detect an innermost loop whose
   accumulator address *varies* in the innermost IV but is *invariant* in an
   enclosing loop (the reduction). Interchange (`affine::interchangeLoops`,
   guarded by `isValidLoopInterchangePermutation`) so the reduction becomes
   innermost → canonical i-j-k.
2. **Imperfect-nest-aware spatial-loop detection**: the two loops indexing the
   accumulator (`enclosingSpatial`) are found by walking parents, *not* by
   requiring a perfect band — so the outer spatial loop `i` may carry the
   beta-scaling sibling. `loopUnrollJamByFactor` already jams imperfect nests.

**Result (N=1024, Ryzen 7950X3D core 8, alpha=1.25 beta=0.75, checksums identical):**
| | GFLOPs |
|---|---|
| PolyBench gemm naive (i-k-j; clang auto-vectorizes the cache-friendly form) | 20.6 |
| **PolyBench gemm + `affine-register-block`** | **50.3** |

**2.4× over clang's own vectorizer on the cache-friendly baseline** — and the
beta-scaling loop is preserved. (Note: i-k-j naive is 20.6, not the 0.47 of the
i-j-k naive — clang vectorizes the stride-1 inner j well; register-blocking still
wins 2.4×.)

## (b) Cache tiling — recover large-N

**Problem.** Register blocking attacks *compute*; it does nothing for *bandwidth*.
Once the matrices exceed the last-level cache the full B is re-streamed per
i-block and the kernel goes DRAM-bound:

| N | working set | register-block-only |
|---|---|---|
| 1024 | 25 MB (L3-resident) | ~52 GFLOPs |
| 2048 | 100 MB (DRAM) | **~12 GFLOPs** (collapse) |

**Fix.** New `cache-tile` option: before register blocking, tile each perfect
GEMM band by `mc x nc x kc` (`affine::tilePerfectlyNested`); the register-block
micro-kernel then runs on the cache-resident point loops. The cache-tile sizes
are a genuine cache-size-dependent knob (unlike the register tile, which the
cost-model spike proved degenerate) — exposed as `mc/nc/kc`, default 256³.

**Result (N=2048, core 8, checksums identical):**
| | GFLOPs |
|---|---|
| register-block only | 13.5 |
| **register-block + `cache-tile` (256³)** | **47.1** |

**3.5× recovery at large N**, back to within ~10% of the L3-resident number. At
N=1024 cache-tile is neutral-to-positive (~50). A small tile-size sweep at N=2048
(spike): 256³ = 45.8, 256×512×256 = 44.0, 128×256×256 = 40.5, 64×128×512 = 34.6
— 256³ best on this machine.

### Narrative this produces
A clean separation of concerns: **cache tiles** are a real, cache-size-dependent
decision (candidate for the existing `dr-affine-loop-tile` cost model — the one
place a cost model has a lever), while the **register tile** is a fixed knob the
spike proved near-invariant. Compose: cache-block for bandwidth + register-block
for compute.

## (b+) Imperfect-nest cache tiling — via loop distribution

`tilePerfectlyNested` needs a perfect band, but the PolyBench `i`-loop carries
the beta-scaling sibling. Rather than the fiddly imperfect strip-mine-and-sink
`tile()`, the pass now **distributes (fissions)** the imperfect outer loop
(`distributeLoop`): `for i { beta_j; matmul }` → `for i { beta_j }` ; `for i { matmul }`.
This is legal — the only shared memref is C, and emitting all-beta-rows then
all-matmul-rows preserves every per-row C dependence (`beta(i)` still precedes
`matmul(i)`). The matmul `i`-loop is then a perfect band and is cache-tiled +
register-blocked as usual; the beta loop is left as its own (untiled, O(N²)) sibling.

> **Bug caught + fixed during bring-up:** the first `distributeLoop` left the
> rewriter's insertion point inside the first new loop's body, so the second
> distributed loop was *nested inside* the first → N× redundant matmul, wrong
> result (0.02 GFLOPs, checksum off). Resetting the insertion point to before the
> original loop each iteration fixed it. Lesson: always verify a structural rewrite
> by **checksum**, not just "valid IR".

**Result — real PolyBench gemm (imperfect i-k-j nest), core 8, checksums identical:**
| N | naive | register-block only | **+ cache-tile (256³)** |
|---|---|---|---|
| 1024 | 20.8 | ~50 | **50.1** |
| 2048 | 9.1 | 28.3 | **48.1** |

At N=2048 the imperfect-nest path now reaches **48.1 GFLOPs — 5.3× over naive,
1.7× over register-block-only** — i.e. the large-N collapse is recovered for the
*real* PolyBench shape, not just the clean perfect nest.

## Validation
- Lit tests (all PASS): `gemm-register-block.mlir` (clean), `gemm-ikj-imperfect.mlir`
  (PolyBench interchange + preserved beta loop), `gemm-cache-tile.mlir` (perfect-nest
  tile + micro-kernel), `gemm-imperfect-cache-tile.mlir` (distribution → tile + block,
  beta sibling preserved). Full suite 185 pass / 16 pre-existing-ONNX unresolved, no regression.

## Honest limits
- Distribution handles an outer loop whose body is a clean **sequence of loops**
  (the faithful PolyBench shape). A stray non-loop op straddling the children
  (some `cgeist -O2` outputs hoist an `index_cast` between the loops) makes
  `distributeLoop` bail → falls back to register-block-only. Replicating such ops
  per child is the next robustness step.
- Fixed f64, single reduction per nest, single matmul band per function tested.
- Tile sizes fixed per invocation (no auto cache-size derivation yet).
- Still single-threaded, single-core; no packing/prefetch (true BLIS ceiling unmeasured).

## Reproduce
- `affine-register-block{mr=8 nr=16 cache-tile=true mc=256 nc=256 kc=256}`
- Harnesses: `/tmp/claude/pb_gemm.c` + `pb_main.c` (PolyBench shape),
  `gemm.mlir.tmpl` + `gemm_main.c` (clean), all run on core 8 `-march=native -ffast-math`.
