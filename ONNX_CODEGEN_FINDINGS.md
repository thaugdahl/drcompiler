# ONNX_CODEGEN_FINDINGS — campaign results

Companion to ONNX_CODEGEN_SPEC.md. Spike-first, honest verdicts. One local
commit per WP (`onnx_codegen_N`). NEVER pushed.

---

## WP-O1 — Reduction de-promotion (the keystone): **WIN**, committed `onnx_codegen_1`

### Spike (go/no-go for the whole campaign): PASS, decisively

Hand-built a 1x1-conv GEMM (M=oc=256, N=p=196, K=ic=256) in onnx-mlir's exact
emitted shape (SSA `affine.for ... iter_args`, batch trip-1 promoted to const 0)
and three lowerings, timed (best-of-5, REPS=2000, marco LLVM 22 `clang -O2`):

| variant | time | vs onnx-mlir original |
|---|---|---|
| iter_args original (onnx-mlir form) | 12.40 s | 1.00x |
| de-promoted, NO register-block (LLVM SLP) | 13.30 s | 0.93x (slower) |
| **de-promoted + register-block vector kernel** | **1.31 s** | **9.4x** |

All 15 runs checksum-identical (`-154.283519`) → bit-correct.
**Risk R1 killed**: LLVM does NOT already vectorize the iter_args inner-product
form, and de-promotion *alone* does nothing — only the explicit broadcast vector
micro-kernel wins, and it wins ~9.4x on a layer class that is ~45-50% of
resnet50's conv FLOPs.

### The design fact the spike forced

A naive in-place de-promotion (`store 0, %C` then `for k { load %C; fma; store
%C }`) reaches only the SCALAR register-block path (mr*nr scalar accumulators +
LLVM SLP), 0 vector ops. Cause: the vectorizer's perfect-body check
(`vectorizeBroadcastBand`, Vectorize.cpp) requires the inner-spatial loop body
to contain ONLY the reduction loop + pure ops; a leading `affine.store` (the
zero-init) has a memory effect and bails it. **The zero-init must live in its
OWN spatial nest** (the cgeist -O0 shape, where `%C` is pre-zeroed elsewhere).
`dr-affine-loop-distribute` does NOT do this fission (it needs ≥2 sibling
*loops*; an init-store + one loop isn't that), so the demote pass fissions the
init nest itself.

### The pass: `dr-scalar-reduction-demote`

`lib/Transforms/ScalarReductionDemote.cpp` (+ td/header/CMake). v1 contract:
single-level f32/f64 `addf` reduction, 1 iter_arg, result stored DIRECTLY,
accumulator address independent of the reduction IV, enclosing spatial band
perfect. Emits: (1) a fresh init nest mirroring the spatial loops that stores
the seed at the accumulator subscript; (2) the reduction band rewritten to a
memref accumulator (load %C / fma / store %C, no iter_args). Bails (leaves
intact) on: nested conv reductions (3x3 — inner addf feeds an outer yield, not a
store), `maxnumf` pool/softmax chains, bias-epilogue stores (Case B, deferred),
multiple iter_args. lit: `test/DataRecomputation/scalar-reduction-demote.mlir`.

### resnet50 coverage

Demotes **33 of 93** iter_args reductions (the single-level 1x1 GEMMs); the
remaining 60 are nested 3x3-conv reductions (WP-O2) + `maxnumf` pools (out of
scope) — all correctly left intact. Standalone correctness re-verified through
the pass-driven pipeline: 12.33 s → 1.31 s, checksum-identical.

---

## Per-band family selection (WP-O4.1, landed with O1 — it was load-bearing)

### Bug found while wiring O1 onto real resnet50

After demote, register-block gave **0 vector ops on resnet50** even though 33
perfect GEMM bands were present. Root cause: `familySelect` in
AffineRegisterBlock.cpp sets a single function-wide `anyDot` flag — if ANY
reduction in the function is classified Dot (rank-k), the WHOLE function switches
to dot mode (scalar 4x4 tile + reassoc), disabling the broadcast vector kernel
for every band. One PolyBench kernel = one family, so the global flag was fine
there; onnx-mlir emits 33+ contractions per function, and a few mis/genuinely
classify as Dot → all 33 fell to the scalar tile. (`family-select=false` forced
broadcast and vectorized all 33: 264 broadcasts — confirming the diagnosis.)

### Fix: per-band family in Stage 3

Family is now detected per band (from the reduction under each `sOut`, before
jamming) and drives that band's jam factor + broadcast-vs-dot kernel choice. For
a single-kernel function this is identical to the old global decision (all bands
share a family), so **PolyBench is unaffected** — full lit 210 pass / 0 fail,
every register-block test (gemm/syrk-family/bmm/blas2) byte-identical vector
counts. The global mode is retained only for the triangular-peel stages and
function-wide fast-math (no-op / tolerable on onnx; correct on single-kernel
PolyBench).

Result: resnet50 now vectorizes the broadcast main loops of **all 33** demoted
bands (136 `vector.broadcast` = 17 main loops × mr 8), up from 0.

### The "17 of 33" is not a coverage gap (instrumented to be sure)

Tagging every family verdict and every `vectorizeBroadcastBand` exit on resnet50:
**all 33 demoted reductions classify BROADCAST (0 Dot)** — detectFamily is not
misclassifying anything. 17 vectorize their main loop; the other 16 exit at
`trip < VL` with **trip = 4** — these are the **vl-remainder tails of the N=196
layers** (196 mod 16 = 4), which the vectorizer deliberately peels off and
leaves scalar (4 of every 196 columns ≈ 2 % of those layers' work). LLVM masks
that tail. So there is no misclassification and no real coverage loss — every
demoted GEMM's bulk vectorizes. (An earlier draft of this doc wrongly guessed
the 16 were Dot-misclassified channels-last layouts; the instrumentation refuted
it. No axis-selection fix is needed.)

---

## Validation harness (built this session)

- `/tmp/onnx_spike/conv1x1_itargs.mlir` — onnx-mlir iter_args form (the input).
- `/tmp/onnx_spike/driver.c` — C driver via `_mlir_ciface_`, REPS loop,
  checksum guard (anti-DCE + correctness).
- `/tmp/onnx_spike/verify.sh` — lowers (marco LLVM 22:
  lower-affine→…→llvm→`clang -O2`), times orig vs demote+register-block, checks
  checksum equality.
- Correctness criterion here is rel-err / checksum, NOT bit-identical (vector
  reductions reassociate) — but the 1x1 GEMM happens to stay bit-identical at
  this size.
- Real IR: `/tmp/onnx_spike/{mnist,resnet50-v2-7}.affine.mlir` (host dr-opt runs
  with `-allow-unregistered-dialect`: the krnl.global weight constants are
  unregistered but the pass only touches affine ops).

## Next

1. **WP-O0 productionize**: end-to-end resnet50 latency harness (lower the whole
   model + run `run_main_graph`) to measure the real inference win and the
   onnx-mlir `--O3` bar — the standalone-kernel 9.4x is per-layer, not the model.
2. **WP-O2**: direct-conv register-block for the 60 nested 3x3 reductions (the
   other ~half of conv FLOPs) — the largest remaining lever.
